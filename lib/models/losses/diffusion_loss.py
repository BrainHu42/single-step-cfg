from functools import partial
from typing import Optional
from typing import Optional

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

    def _forward_loss(self, outputs_dict):
        """Forward function for loss calculation.
        Args:
            outputs_dict (dict): Outputs of the model used to calculate losses.

        Returns:
            torch.Tensor: Calculated loss.
        """
        loss_input_dict = {
            k: outputs_dict[v]
            for k, v in self.data_info.items()
        }
        loss = self.loss_fn(**loss_input_dict) * 0.5
        return loss


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
                'Cannot parse your arguments passed to this loss module.'
                ' Please check the usage of this module')

        # Predicted GM parameters
        means = output_dict['means']            # (B, G, C, H, W) or (B, *, G, C, H, W)
        logstds = output_dict['logstds']        # (B, 1 or G, 1 or C, 1 or H, 1 or W)
        logweights = output_dict['logweights']  # (B, G, 1, H, W)
        uncond_logweights = output_dict['uncond_logweights']

        # Expand predictions with a candidate axis of length 1
        means = means.unsqueeze(1)            # (B, 1, G, C, H, W)
        logstds = logstds.unsqueeze(1)        # (B, 1, 1/G, 1/C, 1/H, 1/W)
        logweights = logweights.unsqueeze(1)  # (B, 1, G, 1, H, W)
        uncond_logweights = uncond_logweights.unsqueeze(1)

        # 1) Conditional KL in x_t_low-space: candidate axis length 1, prob=1 (log_prob=0)
        x_t_low = output_dict['x_t_low']          # (B, C, H, W)
        cond_target = x_t_low.unsqueeze(1)        # (B, 1, C, H, W)
        cond_log_probs = torch.zeros(cond_target.size(0), cond_target.size(1), device=x_t_low.device, dtype=x_t_low.dtype)
        cond_loss = gm_kl_divergence(
            pred_means=means,
            target=cond_target,
            pred_logstds=logstds,
            pred_logweights=logweights,
            true_log_probs=cond_log_probs,
            reduction='flatmean',
        )  # (B,)

        # 2) Unconditional KL over candidate targets
        if self.p_uncond > 0.0:
            # Expect: x_t (B,...) at t_high, candidate bank x0_bank (N= B by default),
            # and x_t_low_matrix (B,B,...) built at t_low to serve as unconditional targets.
            # Using exact single-step posterior over x0 candidates:
            #   p(j | x_t, t) ∝ p(x_t | x0_j, t) · p(j)
            assert all(k in output_dict for k in ('x_t', 'x_t_low_matrix', 'timesteps', 'num_timesteps')), \
                "Provide 'x_t', 'x_t_low_matrix', 'timesteps', and 'num_timesteps' in outputs_dict."

            # Prefer explicit x0_bank key; fall back to common clean-sample keys if present
            x0_bank = None
            for key in ('x0_bank', 'x_0', 'x0'):
                if key in output_dict:
                    x0_bank = output_dict[key]
                    break
            if x0_bank is None:
                raise KeyError("Missing 'x0_bank' (or 'x_0'/'x0') in outputs_dict for posterior computation.")

            # Compute row-wise posterior log-probs over clean candidates
            uncond_log_probs, _ = self.compute_posterior_x0_given_xt(
                x_t=output_dict['x_t'],
                x0_bank=x0_bank,
                timesteps=output_dict['timesteps'],
                num_timesteps=output_dict['num_timesteps'],
                top_k=None,  # use full candidate set so target aligns as (B,B,...)
            )

            # KL w.r.t. model p(x_t_low | GM params) under q(cand) given by uncond_log_probs
            uncond_loss = gm_kl_divergence(
                pred_means=means,
                target=output_dict['x_t_low_matrix'],
                pred_logstds=logstds,
                pred_logweights=uncond_logweights,
                true_log_probs=uncond_log_probs,
                reduction='flatmean',
            )  # (B,)
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
                loss_cond=reduce_loss(cond_loss * self.weight_scale, self.reduction).item(),
                loss_uncond=reduce_loss(uncond_loss * self.weight_scale, self.reduction).item(),
            )

        return reduce_loss(loss_rescaled, self.reduction)

    @torch.no_grad()
    def compute_posterior_x0_given_xt(self,
                                      x_t: torch.Tensor,
                                      x0_bank: torch.Tensor,
                                      timesteps: torch.Tensor,
                                      num_timesteps: float,
                                      log_prior: Optional[torch.Tensor] = None,
                                      top_k: Optional[int] = None,
                                      temperature: float = 1.0,
                                      sigma_extra: float = 0.0,
                                      eps: float = 1e-12):
        """
        Exact single-step posterior over discrete x0 candidates under the linear forward kernel:
            x_t = alpha x_0 + sigma * eps, eps ~ N(0, I).

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
        - log_probs: (B, K) if top_k else (B, N) row-wise log-softmax
        - eps_hat:   (B, K, C, H, W) if top_k else (B, N, C, H, W) estimated noise per candidate
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
        xt_b = x_t.unsqueeze(1)                          # (B,1,C,H,W)
        x0_cols = x0_bank.unsqueeze(0).expand(B, N, *x0_bank.shape[1:])  # (B,N,...)

        # Residual and noise estimate
        resid = xt_b - alpha * x0_cols                   # (B,N,C,H,W)
        eps_hat = resid / sigma                          # (B,N,C,H,W)

        # Log-likelihood up to constants; include constant for completeness (cancels in softmax)
        quad = (resid * resid).sum(dim=tuple(range(- (x_t.dim() - 1), 0)))  # (B,N)
        log_two_pi = torch.log(torch.tensor(2.0 * torch.pi, dtype=dtype, device=device))
        # Collapse per-row sigma^2 to shape (B,1) to avoid accidental broadcasting over B
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
            top_vals, top_idx = torch.topk(log_probs, k=k, dim=1)            # (B,K)
            batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, k)
            eps_topk = eps_hat[batch_idx, top_idx]                            # (B,K,...)
            return top_vals, eps_topk
        return log_probs, eps_hat
