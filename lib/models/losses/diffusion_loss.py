from functools import partial
from typing import Optional, Tuple

import torch
import torch.distributed as dist

from mmgen.models import MODULES
from mmgen.models.losses.ddpm_loss import DDPMLoss, mse_loss, reduce_loss
from mmgen.models.losses.utils import weighted_loss

from ...core import reduce_mean


@weighted_loss
def gaussian_nll_loss(pred, target, logstd, eps=1e-4):
    inverse_std = torch.exp(-logstd).clamp(max=1 / eps)
    diff_weighted = (pred - target) * inverse_std
    return 0.5 * diff_weighted.square() + logstd


@weighted_loss
def gaussian_mixture_nll_loss(
        pred_means, target, pred_logstds, pred_logweights, eps=1e-4):
    """
    Args:
        pred_means (torch.Tensor): Shape (bs, *, num_gaussians, c, h, w)
        target (torch.Tensor): Shape (bs, *, c, h, w)
        pred_logstds (torch.Tensor): Shape (bs, *, 1 or num_gaussians, 1 or c, 1 or h, 1 or w)
        pred_logweights (torch.Tensor): Shape (bs, *, num_gaussians, 1, h, w)

    Returns:
        torch.Tensor: Shape (bs, *, h, w)
    """
    inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)
    diff_weighted = (pred_means - target.unsqueeze(-4)) * inverse_std
    gaussian_ll = (-0.5 * diff_weighted.square() - pred_logstds).sum(dim=-3)  # (bs, *, num_gaussians, h, w)
    loss = -torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-3), dim=-3)
    return loss


def gm_kl_divergence(
        pred_means: torch.Tensor,
        target: torch.Tensor,
        pred_logstds: torch.Tensor,
        pred_logweights: torch.Tensor,
        true_log_probs: torch.Tensor,
        eps: float = 1e-8,
        reduction: str = 'none'):
    """
    KL(q || p) where:
      - p is a Gaussian Mixture given by (pred_means, pred_logstds, pred_logweights)
      - q is a discrete distribution over candidate targets with log-probs true_log_probs

    Shapes (broadcasted over arbitrary middle dims '*'):
      - pred_means:      (bs, *, G, C, H, W)
      - target:          (bs, *, C, H, W)
      - pred_logstds:    (bs, *, 1 or G, 1 or C, 1 or H, 1 or W)
      - pred_logweights: (bs, *, G, 1, H, W)
      - true_log_probs:  (bs, *)  — along the same '*' candidate axis as target

    Returns:
      - kl: (bs, *, H, W) if reduction='none'
            reduced per reduction otherwise ('mean'|'sum'|'flatmean').
    Notes:
      - H(q) term (−E_q[log q]) is included for completeness but has zero gradient w.r.t model.
    """
    # Compute log p(x | GM)
    # Inverse-std clamp and weighted diffs
    inverse_std = torch.exp(-pred_logstds).clamp(max=1 / eps)
    diff_weighted = (target.unsqueeze(-4) - pred_means) * inverse_std

    # Gaussian per-component log-likelihood:
    #   (-0.5 * sum_D ((x-μ)/σ)^2  - sum_D logσ)
    # Broadcast logstds across the summed dimension so isotropic components
    # correctly subtract D * logstd when reduced over channels.
    gaussian_ll = (
        -0.5 * diff_weighted.square() - pred_logstds
    ).sum(dim=-3)  # (bs, *, G, H, W)
    log_p = torch.logsumexp(gaussian_ll + pred_logweights.squeeze(-3), dim=-3)  # (bs, *, H, W)

    # Broadcast q's log-prob across spatial dims
    log_q = true_log_probs.unsqueeze(-1).unsqueeze(-1)  # (bs, *, 1, 1) -> (bs, *, H, W)
    log_q = log_q.expand_as(log_p)

    # Mask non-finite entries of q (log_q = -inf => prob 0)
    finite = torch.isfinite(log_q)
    log_q_safe = torch.where(finite, log_q, torch.zeros_like(log_q))
    log_p_safe = torch.where(finite, log_p, torch.zeros_like(log_p))
    q = torch.exp(log_q_safe)

    kl = q * (log_q_safe - log_p_safe)  # (bs, *, H, W)

    if reduction == 'none':
        return kl
    elif reduction == 'mean':
        return kl.mean()
    elif reduction == 'sum':
        return kl.sum()
    elif reduction == 'flatmean':
        return kl.flatten(1).mean(dim=1)
    else:
        raise ValueError(f'Unknown reduction: {reduction}')


class DDPMLossMod(DDPMLoss):

    def __init__(self,
                 *args,
                 weight_scale=1,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_scale = weight_scale

    def timestep_weight_rescale(self, loss, timesteps, weight):
        return loss * weight.to(timesteps.device)[timesteps] * self.weight_scale

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = self.rescale_fn(loss, timesteps)

        # update log_vars of this class
        self.collect_log(loss_rescaled, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class DDPMMSELossMod(DDPMLossMod):
    _default_data_info = dict(pred='eps_t_pred', target='noise')

    def __init__(self,
                 rescale_mode=None,
                 rescale_cfg=None,
                 sampler=None,
                 weight=None,
                 weight_scale=1.0,
                 log_cfgs=None,
                 reduction='mean',
                 data_info=None,
                 loss_name='loss_ddpm_mse',
                 scale_norm=False,
                 p_uncond=0.5,
                 num_timesteps=1000,
                 momentum=0.001):
        super().__init__(rescale_mode=rescale_mode,
                         rescale_cfg=rescale_cfg,
                         log_cfgs=log_cfgs,
                         weight=weight,
                         weight_scale=weight_scale,
                         sampler=sampler,
                         reduction=reduction,
                         loss_name=loss_name)

        self.data_info = self._default_data_info \
            if data_info is None else data_info

        self.num_timesteps = num_timesteps
        self.p_uncond = p_uncond
        self.loss_fn = partial(mse_loss, reduction='flatmean')
        self.scale_norm = scale_norm
        self.freeze_norm = False
        if scale_norm:
            self.register_buffer('norm_factor', torch.ones(1, dtype=torch.float))
        self.momentum = momentum

    def forward(self, *args, **kwargs):
        loss = super().forward(*args, **kwargs)
        if self.scale_norm:
            if self.training and not self.freeze_norm:
                if len(args) == 1:
                    assert isinstance(args[0], dict), (
                        'You should offer a dictionary containing network outputs '
                        'for building up computational graph of this loss module.')
                    output_dict = args[0]
                elif 'output_dict' in kwargs:
                    assert len(args) == 0, (
                        'If the outputs dict is given in keyworded arguments, no'
                        ' further non-keyworded arguments should be offered.')
                    output_dict = kwargs.pop('outputs_dict')
                else:
                    raise NotImplementedError(
                        'Cannot parsing your arguments passed to this loss module.'
                        ' Please check the usage of this module')
                norm_factor = output_dict['x_0'].detach().square().mean()
                norm_factor = reduce_mean(norm_factor)
                self.norm_factor[:] = (1 - self.momentum) * self.norm_factor \
                                      + self.momentum * norm_factor
            loss = loss / self.norm_factor
        return loss

    def _forward_loss(self, output_dict):
        """Compute loss.
        If Gaussian-mixture keys are present, compute both conditional and
        unconditional u_t losses as requested; otherwise, fall back to
        vanilla MSE between ``pred`` and ``target`` from ``data_info``.
        """
        has_gm = all(k in output_dict for k in ("means", "logweights", "logstds"))
        if not has_gm:
            # CHANGED: remove 0.5; match forward's "no scaling", and rescale on return.
            loss_input_dict = {k: output_dict[v] for k, v in self.data_info.items()}
            pred = loss_input_dict.get('pred')
            target = loss_input_dict.get('target')
            if isinstance(pred, torch.Tensor) and isinstance(target, torch.Tensor):
                if pred.shape != target.shape and pred.dim() > target.dim():
                    squeezed = pred
                    while squeezed.dim() > target.dim():
                        dim_to_squeeze = next((i for i, size in enumerate(squeezed.shape)
                                                if size == 1), None)
                        if dim_to_squeeze is None:
                            break
                        squeezed = squeezed.squeeze(dim_to_squeeze)
                    if squeezed.shape == target.shape:
                        loss_input_dict['pred'] = squeezed
            loss = self.loss_fn(**loss_input_dict) * 0.5
            return reduce_loss(loss * self.weight_scale, self.reduction)


        # 1) Conditional u_t prediction from mixture mean
        means = output_dict["means"]  # (B,*,G,C,H,W) or (B,G,C,H,W)
        logweights = output_dict["logweights"]
        # normalize dims and compute softmax over Gaussian axis (-4)
        while logweights.dim() < means.dim():
            logweights = logweights.unsqueeze(-1)
        w = torch.softmax(logweights, dim=-4)
        u_t_pred = (means * w).sum(dim=-4)  # (B,*,C,H,W)

        # target u_t = noise - x_0 (provided by GaussianFlow) or reconstruct
        if "u_t" in output_dict:
            u_t_tgt = output_dict["u_t"]
        else:
            assert all(k in output_dict for k in ("noise", "x_0")), (
                "Need either 'u_t' or both 'noise' and 'x_0' in outputs_dict")
            u_t_tgt = output_dict["noise"] - output_dict["x_0"]

        # CHANGED: keep a raw part for logging; weight by (1 - p_uncond) only for the total.
        loss_cond_raw = self.loss_fn(pred=u_t_pred, target=u_t_tgt) * 0.5
        loss_cond = loss_cond_raw * (1.0 - self.p_uncond)

        # 2) Unconditional term (if uncond_* present):
        loss_uncond = None
        loss_uncond_raw = None
        has_uncond = "uncond_mean" in output_dict # all(k in outputs_dict for k in ("uncond_means", "uncond_logweights", "uncond_logstds"))
        if has_uncond:
            u_uncond_pred = output_dict["uncond_mean"]

            # Build target: E_{posterior}[ (x0 - x_t)/sigma ] over batch x0 bank
            assert all(k in output_dict for k in ("x_0", "noise", "timesteps", "x_t"))
            x_t = output_dict["x_t"]
            x0_bank = output_dict["x_0"]  # (B,C,H,W) treated as candidate set
            timesteps = output_dict["timesteps"].to(x0_bank.dtype)
            T = float(self.num_timesteps if self.num_timesteps is not None else 1000)

            log_probs, vector_field = compute_posterior_x0_given_xt(
                x_t=x_t,
                x0_bank=x0_bank,
                timesteps=timesteps,
                num_timesteps=T,
            )  # log_probs: (B,B), vector_field: (B,B,C,H,W)

            probs = log_probs.softmax(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            u_uncond_tgt = (probs * vector_field).sum(dim=1)  # (B,C,H,W)

            # CHANGED: keep a raw part for logging; weight by p_uncond only for the total.
            loss_uncond_raw = self.loss_fn(pred=u_uncond_pred, target=u_uncond_tgt) * 0.5
            loss_uncond = loss_uncond_raw * self.p_uncond

        # Total loss: sum available parts (weighted by p_uncond)
        loss = loss_cond if loss_uncond is None else (loss_cond + loss_uncond)

        # Expose parts for logging: match forward() by logging the RAW parts times weight_scale
        to_log = dict(
            loss_cond=reduce_loss(loss_cond_raw, self.reduction).item()
        )
        if loss_uncond_raw is not None:
            to_log["loss_uncond"] = reduce_loss(loss_uncond_raw, self.reduction).item()
        self.log_vars.update(**to_log)

        return reduce_loss(loss, self.reduction)


@MODULES.register_module()
class FlowNLLLoss(DDPMLossMod):
    _default_data_info = dict(pred='u_t_pred', target='u_t', logstd='logstd')

    def __init__(self,
                 weight_scale=1.0,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_ddpm_nll'):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1, ), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1, ), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1, ), dtype=torch.long))

    @torch.no_grad()
    def collect_log(self, loss, var, timesteps):
        if not self.log_fn_list:
            return

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder_l = [torch.zeros_like(loss) for _ in range(ws)]
            placeholder_v = [torch.zeros_like(var) for _ in range(ws)]
            placeholder_t = [torch.zeros_like(timesteps) for _ in range(ws)]
            dist.all_gather(placeholder_l, loss)
            dist.all_gather(placeholder_v, var)
            dist.all_gather(placeholder_t, timesteps)
            loss = torch.cat(placeholder_l, dim=0)
            var = torch.cat(placeholder_v, dim=0)
            timesteps = torch.cat(placeholder_t, dim=0)
        log_vars = dict()

        if (dist.is_initialized()
                and dist.get_rank() == 0) or not dist.is_initialized():
            for log_fn in self.log_fn_list:
                log_vars.update(log_fn(loss, var, timesteps))
        self.log_vars = log_vars

    @torch.no_grad()
    def quartile_log_collect(self,
                             loss,
                             var,
                             timesteps,
                             total_timesteps,
                             prefix_name,
                             reduction='mean',
                             momentum=0.1):
        quartile = (timesteps / total_timesteps * 4)
        quartile = quartile.to(torch.long).clamp(min=0, max=3)

        log_vars = dict()

        for idx in range(4):
            quartile_mask = quartile == idx
            quartile_count = torch.count_nonzero(quartile_mask).reshape(1)
            if quartile_count > 0:
                loss_quartile = reduce_loss(loss[quartile_mask], reduction).reshape(1)
                var_quartile = reduce_loss(var[quartile_mask], reduction).reshape(1)

                cur_weight = 1 - torch.exp(-momentum * quartile_count)
                getattr(self, f'count_quartile_{idx}').add_(quartile_count)
                total_weight = 1 - torch.exp(-momentum * getattr(self, f'count_quartile_{idx}'))
                cur_weight /= total_weight.clamp(min=1e-4)
                getattr(self, f'loss_quartile_{idx}').mul_(1 - cur_weight).add_(loss_quartile * cur_weight)
                getattr(self, f'var_quartile_{idx}').mul_(1 - cur_weight).add_(var_quartile * cur_weight)

            log_vars[f'{prefix_name}_quartile_{idx}'] = getattr(self, f'loss_quartile_{idx}').item()
            log_vars[f'{prefix_name}_var_quartile_{idx}'] = getattr(self, f'var_quartile_{idx}').item()

        return log_vars

    def _forward_loss(self, outputs_dict):
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict)
        return loss

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = loss * self.weight_scale

        with torch.no_grad():
            var = torch.exp(output_dict['logstd'] * 2)  # (bs, *)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']]  # (bs, *)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowNLLLoss(FlowNLLLoss):
    _default_data_info = dict(
        pred_means='means',
        target='u_t',
        pred_logstds='logstds',
        pred_logweights='logweights')

    def __init__(self,
                 weight_scale=1.0,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_ddpm_nll'):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_mixture_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                'You should offer a dictionary containing network outputs '
                'for building up computational graph of this loss module.')
            output_dict = args[0]
        elif 'output_dict' in kwargs:
            assert len(args) == 0, (
                'If the outputs dict is given in keyworded arguments, no'
                ' further non-keyworded arguments should be offered.')
            output_dict = kwargs.pop('outputs_dict')
        else:
            raise NotImplementedError(
                'Cannot parsing your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # check keys in output_dict
        assert 'timesteps' in output_dict, (
            '\'timesteps\' is must for DDPM-based losses, but found'
            f'{output_dict.keys()} in \'output_dict\'')

        timesteps = output_dict['timesteps']
        loss = self._forward_loss(output_dict)

        loss_rescaled = loss * self.weight_scale

        with torch.no_grad():
            weights = output_dict['logweights'].exp()
            mean = (weights * output_dict['means']).sum(-4, keepdim=True)  # (bs, *, 1, c, h, w)
            var = (weights * ((output_dict['means'] - mean).square()
                              + (output_dict['logstds'] * 2).exp())).sum(-4)  # (bs, *, c, h, w)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']].unsqueeze(-3)  # (bs, *, 1, h, w)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss

            # update log_vars of this class
            self.collect_log(_loss, _var, timesteps=timesteps)  # Mod: log after rescaling

        return reduce_loss(loss_rescaled, self.reduction)


@MODULES.register_module()
class GMFlowHybridLoss(FlowNLLLoss):
    """
    Hybrid loss combining:
    - Conditional NLL on the observed pair (u_t = noise - x_0), as in GMFlowNLLLoss.
    - Unconditional, pair-weighted NLL over candidate targets using uncond_ut and uncond_row_probs.

    Expects the following keys in outputs_dict in addition to GMFlowNLLLoss keys:
    - 'uncond_ut': Tensor (B, B, C, H, W)
    - 'uncond_row_probs' or 'uncond_probs': probabilities or log-likelihoods over the second B dimension

    Args:
        p_uncond (float): Mixture weight for the unconditional term.
    """

    def __init__(self,
                 weight_scale=1.0,
                 p_uncond=0.2,
                 top_k=None,
                 log_cfgs=None,
                 data_info=None,
                 reduction='mean',
                 loss_name='loss_gmflow_hybrid'):
        super().__init__(
            weight_scale=weight_scale,
            log_cfgs=log_cfgs,
            reduction=reduction,
            loss_name=loss_name)
        self.p_uncond = float(p_uncond)
        self.top_k = top_k
        self.data_info = self._default_data_info \
            if data_info is None else data_info
        self.loss_fn = partial(gaussian_mixture_nll_loss, reduction='flatmean')
        if log_cfgs is not None and log_cfgs.get('type', None) == 'quartile':
            for i in range(4):
                self.register_buffer(f'loss_quartile_{i}', torch.zeros((1,), dtype=torch.float))
                self.register_buffer(f'var_quartile_{i}', torch.ones((1,), dtype=torch.float))
                self.register_buffer(f'count_quartile_{i}', torch.zeros((1,), dtype=torch.long))

    @staticmethod
    def _ensure_K_axis(x: torch.Tensor, want_ndim: int = 6, name: str = "tensor") -> torch.Tensor:
        """
        Ensure x has a candidate axis K at dim=1. Target is (B,K,G,C,H,W).
        Accepts (B,G,C,H,W) or (B,K,G,C,H,W).
        """
        if x.dim() == want_ndim:
            return x
        if x.dim() == want_ndim - 1:
            return x.unsqueeze(1)
        raise ValueError(f"{name}: expected {want_ndim-1}D or {want_ndim}D, got {tuple(x.shape)}")


    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Parse output dict
        if len(args) == 1:
            assert isinstance(args[0], dict), (
                "You should pass a dictionary containing network outputs "
                "to build the computational graph for this loss module."
            )
            output_dict = args[0]
        elif "output_dict" in kwargs:
            assert len(args) == 0, (
                "If the outputs dict is given as a keyword argument, "
                "no positional arguments should be provided."
            )
            output_dict = kwargs.pop("output_dict")  # FIXED: correct key
        else:
            raise NotImplementedError(
                "Cannot parse your arguments passed to this loss module. "
                "Please call with either loss(output_dict) or loss(output_dict=...)."
            )

        # Read new-API split head predictions
        means      = output_dict["means"]        # (B,G,C,H,W) or (B,K,G,C,H,W)
        logstds    = output_dict["logstds"]      # broadcastable to (B,K,G,C,H,W)
        logweights = output_dict["logweights"]   # (B,G,1,H,W) or (B,K,G,1,H,W)

        uncond_mean = output_dict["uncond_mean"]

        # Normalize shapes to (B,K,G,C,H,W)
        means      = self._ensure_K_axis(means,      name="means")
        logstds    = self._ensure_K_axis(logstds,    name="logstds")
        logweights = self._ensure_K_axis(logweights, name="logweights")

        # 1) Conditional KL in x_t_low-space (K=1)
        x_t_low = output_dict["x_t_low"]               # (B,C,H,W)
        cond_target = x_t_low.unsqueeze(1)             # (B,1,C,H,W)
        cond_log_probs = torch.zeros(
            (cond_target.size(0), cond_target.size(1)),
            device=x_t_low.device,
            dtype=x_t_low.dtype,
        )  # (B,1) -> log(1.0)

        cond_loss = gm_kl_divergence(
            target=cond_target,               # (B,1,C,H,W)
            true_log_probs=cond_log_probs,    # (B,1)
            pred_means=means,                 # (B,1,G,C,H,W)
            pred_logstds=logstds,             # (B,1,[G|1],[C|1],[H|1],[W|1])
            pred_logweights=logweights,       # (B,1,G,1,H,W)
            reduction="flatmean",
        )  # (B,)

        # 2) Unconditional KL over candidate targets
        if self.p_uncond > 0.0:
            required = ("x_t", "timesteps", "t_low", "num_timesteps", "x_0")
            missing = [k for k in required if k not in output_dict]
            assert not missing, f"Missing keys for uncond loss: {missing}"

            # Compute posterior over candidates x_{t_low}
            uncond_log_probs, velocity_fields = compute_posterior_x0_given_xt(
                x_t=output_dict['x_t'],
                x0_bank=output_dict['x_0'],
                timesteps=output_dict['timesteps'],
                num_timesteps=output_dict['num_timesteps'],
            )

            # Prepare predicted unconditional mean (B,C,H,W)
            if uncond_mean.dim() == 4:
                uncond_pred = uncond_mean
            elif uncond_mean.dim() == 5:
                if uncond_mean.size(1) == 1:
                    uncond_pred = uncond_mean.squeeze(1)
                else:
                    raise ValueError(
                        f"uncond_mean has incompatible K dimension {uncond_mean.size(1)}; expected 1 or absent")
            else:
                raise ValueError(f"uncond_mean must be 4D or 5D, got shape {tuple(uncond_mean.shape)}")

            # Compute unconditional target as expectation over velocity fields
            weights = uncond_log_probs.softmax(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            uncond_target = (weights * velocity_fields).sum(dim=1)

            diff = uncond_pred - uncond_target
            uncond_loss = diff.flatten(1).square().mean(dim=1)
        else:
            uncond_loss = torch.zeros_like(cond_loss)

        # Mixture
        loss = (1.0 - self.p_uncond) * cond_loss + self.p_uncond * uncond_loss  # (B,)

        # Rescale
        loss_rescaled = loss * self.weight_scale

        # Record combined loss/variance like other losses, and also log parts
        with torch.no_grad():
            timesteps = output_dict['timesteps']
            weights = logweights.exp()
            mean = (weights * means).sum(-4, keepdim=True)  # (B, *, 1, C, H, W)
            var = (weights * ((means - mean).square() + (logstds * 2).exp())).sum(-4)  # (B, *, C, H, W)
            if 'weight' in self.data_info:
                weight = output_dict[self.data_info['weight']].unsqueeze(-3)  # (B, *, 1, H, W)
                weight_norm_factor = weight.flatten(1).mean(dim=1).clamp(min=1e-6)
                _var = (var * weight).flatten(1).mean(dim=1) / weight_norm_factor
                _loss = loss / weight_norm_factor
            else:
                _var = var.flatten(1).mean(dim=1)
                _loss = loss
            # Collect fused logs
            self.collect_log(_loss, _var, timesteps=timesteps)
            # Append separate parts
            self.log_vars.update(
                loss_cond=reduce_loss(cond_loss, self.reduction).item(),
                loss_uncond=reduce_loss(uncond_loss, self.reduction).item(),
            )

        return reduce_loss(loss_rescaled, self.reduction)

@torch.no_grad()
def compute_posterior_x0_given_xt(
    x_t: torch.Tensor,
    x0_bank: torch.Tensor,
    timesteps: torch.Tensor,
    num_timesteps: float,
    log_prior: Optional[torch.Tensor] = None,
    top_k: Optional[int] = None,
    temperature: float = 1.0,
    sigma_extra: float = 0.0,
    eps: float = 1e-12,
):
    """
    Exact single-step posterior over discrete x0 candidates under the linear forward kernel:
        x_t = alpha x_0 + sigma * eps,  eps ~ N(0, I), with alpha = 1 - t/T, sigma = t/T.

    p(j | x_t, t) ∝ p(x_t | x0_j, t) · p(j) with
        log p(x_t | x0_j, t) = −0.5 / sigma^2 · ||x_t − alpha x0_j||^2 − 0.5 D log(2π sigma^2).

    Inputs
    - x_t: (B, C, H, W)
    - x0_bank: (N, C, H, W) or (B, C, H, W) interpreted as N=B
    - timesteps: (B,) integer or float in [0, T]
    - num_timesteps: scalar T
    - log_prior: optional (N,) or (B, N) log p(j); defaults to uniform
    - top_k: optional int; return only Top-K per row
    - temperature: softmax temperature τ; τ>1 smooths
    - sigma_extra: extra noise std added in quadrature for smoothing/mismatch

    Returns
    - log_probs:   (B, K) if top_k else (B, N) row-wise log-softmax
    - vector_field:(B, K, C, H, W) if top_k else (B, N, C, H, W) where
                vector_field[i, j] = (x0_j − x_t[i]) / sigma(t[i])

    Note: This matches the vector field from `compute_posterior_x1_given_xt`
            under the identification x1 ≡ x0 and t(second)=alpha=1−t/T.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    if sigma_extra < 0:
        raise ValueError(f"sigma_extra must be >= 0, got {sigma_extra}")

    B = x_t.size(0)
    D = x_t[0].numel()
    device = x_t.device
    dtype = x_t.dtype

    # Candidate axis length N
    if x0_bank.dim() != x_t.dim():
        raise ValueError(f"x0_bank must have same dims as x_t; got {x0_bank.shape} vs {x_t.shape}")
    N = x0_bank.size(0)

    # Compute alpha_t and sigma_t
    T = float(num_timesteps)
    t = timesteps.to(torch.float32)  # (B,)
    alpha = (1.0 - t / T).to(dtype).view(B, 1, *([1] * (x_t.dim() - 1)))  # (B,1,1,1)
    sigma = (t / T).to(dtype).view(B, 1, *([1] * (x_t.dim() - 1))).clamp_min(eps)
    sigma2_eff = sigma.square() + (torch.tensor(float(sigma_extra), dtype=dtype, device=device) ** 2)

    # Broadcast x_t and candidates
    xt_b = x_t.unsqueeze(1)  # (B,1,C,H,W)
    x0_cols = x0_bank.unsqueeze(0).expand(B, N, *x0_bank.shape[1:])  # (B,N,...)

    # Residual and noise estimate
    resid = xt_b - alpha * x0_cols         # (B,N,C,H,W)
    eps_hat = resid / sigma                # (B,N,C,H,W)

    # === NEW: vector field matching compute_posterior_x1_given_xt ===
    # (x0 - x_t) / sigma  ==  x0 - eps_hat   because alpha + sigma = 1
    vector_field = x0_cols - eps_hat       # (B,N,C,H,W)

    # Log-likelihood (row-wise constants included; cancel in softmax)
    quad = (resid * resid).sum(dim=tuple(range(- (x_t.dim() - 1), 0)))  # (B,N)
    log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
    sigma2_row = sigma2_eff.view(B, -1)[:, :1]                           # (B,1)
    log_const = 0.5 * D * (torch.log(sigma2_row) + log_two_pi)           # (B,1)
    logits = -(0.5 * quad / sigma2_row) - log_const                      # (B,N)

    # Add log prior if provided
    if log_prior is not None:
        if log_prior.dim() == 1:
            if log_prior.size(0) != N:
                raise ValueError(f"log_prior length {log_prior.size(0)} != N={N}")
            logits = logits + log_prior.view(1, N)
        elif log_prior.dim() == 2:
            if log_prior.shape != logits.shape:
                raise ValueError(f"log_prior shape {tuple(log_prior.shape)} != {tuple(logits.shape)}")
            logits = logits + log_prior
        else:
            raise ValueError("log_prior must be (N,) or (B,N)")

    # Temperature smoothing and normalization
    log_probs = torch.log_softmax(logits / float(temperature), dim=1)     # (B,N)

    if top_k is not None:
        k = int(top_k)
        if k <= 0:
            raise ValueError(f"top_k must be positive, got {k}")
        k = min(k, N)
        top_vals, top_idx = torch.topk(log_probs, k=k, dim=1)             # (B,K)
        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
        vf_topk = vector_field[batch_idx, top_idx]                         # (B,K,...)
        return top_vals, vf_topk

    return log_probs, vector_field
