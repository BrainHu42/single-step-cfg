#!/usr/bin/env python3
"""
Visualize the unconditional row-wise probabilities as a heatmap.

What it shows
- A B×B matrix of exp(uncond_log_probs) computed by
  GMFlowHybridLoss.compute_prior_log_likelihood using a training-like pair:
  rows at a low timestep t_low and x_t_high at a higher timestep t_high.
- Row i corresponds to a fixed row noise/noise_0[i] and t_low[i];
  columns j correspond to candidate clean samples. The pairwise matrix
  x_t_low_matrix[i, j] = (1 − t_low[i]/T)·x0[j] + (t_low[i]/T)·noise_0[i].

Modes
- dataset (default): builds a small batch from a dataset (checkerboard or
  ImageNet), constructs (x_t_low, x_t_high, x_t_low_matrix, t_low, t_high)
  the same way as GMFlowSSCFG.forward_train/.loss, computes uncond_log_probs,
  and plots exp(logp).
- file: loads a saved matrix (e.g., uncond_log_probs) and plots its exp.

Key arguments
- --batch-size B: batch size used to form the B×B matrix.
- --num-timesteps T: diffusion horizon used to normalize timesteps.
- --trans-ratio r: sets t_low = t_high · (1 − r). Use r in (0, 1].
- --eps: small epsilon to ensure t_low < t_high.
- --top-k: optional; if set, keep top‑K columns per row when computing.
- --tau: optional in [0,1]; if set, forces all t_high = round(tau·T).
- For ImageNet: --imagenet-root, --datalist, --label2name, --image-size.
- --save: path to save the heatmap image; otherwise shows an interactive window.

Examples
- Checkerboard 2D (dataset mode):
  python scripts/visualize_uncond_probs.py --mode dataset --dataset checkerboard \
      --batch-size 128 --num-timesteps 1000 --trans-ratio 1.0 --save heatmap.png

- ImageNet (dataset mode):
  python scripts/visualize_uncond_probs.py --mode dataset --dataset imagenet \
      --imagenet-root data/imagenet/train_cache --datalist data/imagenet/train_cache.txt \
      --label2name data/imagenet/imagenet1000_clsidx_to_labels.txt \
      --batch-size 256 --image-size 256 --num-timesteps 1000 --trans-ratio 0.5 \
      --save heatmap.png --tau 1.0

- File mode (.pt/.pth/.npy/.npz):
  python scripts/visualize_uncond_probs.py --mode file your_tensor.pth --key uncond_log_probs
"""

import argparse
import pathlib
from typing import Any, Optional

import numpy as np
import os
import sys


def _load_array(path: str, key: Optional[str]) -> np.ndarray:
    p = pathlib.Path(path)
    suf = p.suffix.lower()
    if suf in {".pt", ".pth", ".pth.tar"}:
        import torch  # lazy import
        obj: Any = torch.load(path, map_location="cpu")
        tensor = None
        if hasattr(obj, "shape") and hasattr(obj, "detach"):
            tensor = obj  # direct Tensor
        elif isinstance(obj, dict):
            # prefer explicit key
            if key is not None and key in obj:
                tensor = obj[key]
            else:
                # best-effort common names
                for k in ("uncond_log_probs", "log_probs", "uncond", "arr"):
                    if k in obj:
                        tensor = obj[k]
                        break
                if tensor is None:
                    # fallback: first tensor-like value
                    for v in obj.values():
                        if hasattr(v, "shape") and hasattr(v, "detach"):
                            tensor = v
                            break
        if tensor is None:
            raise ValueError("Could not locate Tensor in the provided file.")
        arr = tensor.detach().cpu().float().exp().numpy()
        return arr

    if suf == ".npy":
        arr = np.load(path)
        return np.exp(arr)

    if suf == ".npz":
        z = np.load(path)
        if key is None:
            # pick first array if no key given
            key = list(z.keys())[0]
        return np.exp(z[key])

    raise ValueError(f"Unsupported file type: {suf}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize uncond_log_probs.exp() as heatmap")
    ap.add_argument("path", nargs="?", help="Path to .pt/.pth/.npy/.npz file (for --mode file)")
    ap.add_argument("--mode", choices=["dataset", "file"], default="dataset", help="Run mode")
    ap.add_argument("--key", default=None, help="Dict/npz key (if needed in file mode)")
    ap.add_argument("--save", default=None, help="Output image path to save (e.g. heatmap.png)")
    ap.add_argument("--cmap", default="viridis", help="Matplotlib colormap name")
    ap.add_argument("--title", default="row_probs.exp()", help="Figure title")

    # Dataset-mode options
    ap.add_argument("--batch-size", type=int, default=128, help="Batch size B for one step")
    ap.add_argument("--num-timesteps", type=int, default=1000, help="Diffusion horizon T")
    ap.add_argument("--trans-ratio", type=float, default=1.0, help="Transition ratio used to set t_low = t_high * (1 - trans_ratio)")
    ap.add_argument("--eps", type=float, default=1e-4, help="Small epsilon to keep t_low < t_high and >= 0")
    ap.add_argument("--top-k", type=int, default=None, help="If set, use Top‑K per row")
    ap.add_argument("--tau", type=float, default=None, help="Fix timestep as tau in [0,1]; overrides random t")
    ap.add_argument("--post-temperature", type=float, default=1.0, help="Posterior temperature (>1 smooths)")
    ap.add_argument("--post-sigma-extra", type=float, default=0.0, help="Extra noise std added in quadrature")
    ap.add_argument("--dataset", choices=["checkerboard", "imagenet"], default="checkerboard",
                    help="Dataset to use in dataset mode")
    # ImageNet-specific
    ap.add_argument("--imagenet-root", default="data/imagenet/train", help="ImageNet data root")
    ap.add_argument("--datalist", default="data/imagenet/train.txt", help="Path to train.txt list")
    ap.add_argument("--label2name", default="data/imagenet/imagenet1000_clsidx_to_labels.txt",
                    help="Path to label2name mapping txt")
    ap.add_argument("--image-size", type=int, default=256, help="Image resolution for preprocessing")
    ap.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 = main process)")
    args = ap.parse_args()

    if args.mode == "file":
        if not args.path:
            raise SystemExit("Provide a file path in file mode.")
        mat = _load_array(args.path, args.key)
    else:
        # dataset mode: produce uncond_log_probs from a tiny batch
        # Ensure repo root on sys.path to import local modules
        this_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(this_dir, ".."))
        if repo_root not in sys.path:
            sys.path.insert(0, repo_root)

        import torch
        from lib.datasets import CheckerboardData, ImageNet, build_dataloader
        from lib.models.losses.diffusion_loss import GMFlowHybridLoss, compute_posterior_x0_given_xt

        device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if args.dataset == "checkerboard":
            # 2D toy data
            dataset = CheckerboardData(scale=4, n_samples=max(args.batch_size, 4096))
            x0 = dataset.samples[: args.batch_size].to(device)  # (B, 2)
        else:
            # ImageNet images: build a minimal DataLoader and take one batch
            dataset = ImageNet(
                data_root=args.imagenet_root,
                datalist_path=args.datalist,
                label2name_path=args.label2name,
                image_size=args.image_size,
                random_flip=False,
                test_mode=False
            )
            loader = build_dataloader(
                dataset,
                samples_per_gpu=args.batch_size,
                workers_per_gpu=args.workers,
                num_gpus=1,
                dist=False,
                shuffle=True,
            )
            batch = next(iter(loader))
            if 'images' in batch:
                x0 = batch['images'].to(device)
            elif 'latents' in batch:
                x0 = batch['latents'].to(device)
            else:
                raise RuntimeError("Batch missing 'images' or 'latents'")

        # Sample a training timestep t_h and construct x_t_low at t_l, then x_t_high via forward transition
        T = int(args.num_timesteps)
        B = x0.size(0)
        if args.tau is not None:
            tau = float(args.tau)
            tau = max(0.0, min(1.0, tau))
            tau = torch.full((B,), tau, device=device, dtype=torch.float32)
            timesteps = (tau * float(T)).round().clamp_(min=0, max=T).long()
        else:
            timesteps = torch.randint(1, T + 1, (B,), device=device)
        t_high = timesteps.to(torch.float32)  # (B,)

        # Derive t_low from t_high using trans_ratio, clamp to [0, t_high - eps]
        eps = float(args.eps)
        trans_ratio = float(args.trans_ratio)
        t_low = (t_high * (1.0 - trans_ratio)).clamp(min=0.0)
        t_low = torch.minimum(t_low, (t_high - eps).clamp_min(0.0))  # ensure t_low < t_high

        # Sample noises: noise_0 for x_t_low, noise_1 for transition to x_t_high
        noise = torch.randn(B * 2, *x0.shape[1:], device=device, dtype=x0.dtype)
        noise_0, noise_1 = torch.chunk(noise, 2, dim=0)

        # Compute x_t_low = (1 - sigma_l) * x0 + sigma_l * noise_0, where sigma_l = t_low / T
        sigma_l = (t_low / float(T)).view(B, *([1] * (x0.dim() - 1)))
        mean_l = 1.0 - sigma_l
        x_t_low = mean_l * x0 + sigma_l * noise_0

        # Compute x_t_high via forward transition from (x_t_low, t_low) to t_high with noise_1
        sigma_h = (t_high / float(T)).view(B, *([1] * (x0.dim() - 1)))
        mean_h = 1.0 - sigma_h
        mean_trans = mean_h / mean_l.clamp_min(1e-8)
        std_trans = (sigma_h.square() - (mean_trans * sigma_l).square()).clamp_min(0.0).sqrt()
        x_t_high = mean_trans * x_t_low + std_trans * noise_1

        # Print timesteps for debugging
        try:
            print("[check] t_high:", t_high.detach().cpu().tolist())
            print("[check] t_low:", t_low.detach().cpu().tolist())
        except Exception:
            pass

        # Compute exact posterior over x0 candidates (columns are x0[j])
        loss_mod = GMFlowHybridLoss(p_uncond=1.0, top_k=args.top_k).to(device)
        with torch.no_grad():
            posterior = compute_posterior_x0_given_xt(
                x_t=x_t_high,
                x0_bank=x0,
                timesteps=t_high,
                num_timesteps=float(T),
            )

        import torch as _torch

        if isinstance(posterior, (list, tuple)) and len(posterior) == 2:
            out1, out2 = posterior
        else:
            raise RuntimeError(
                "compute_posterior_x0_given_xt expected to return (log_probs, vector_field)"
            )

        logp = None
        vector_field = None
        for candidate in (out1, out2):
            if not hasattr(candidate, "dim"):
                continue
            if candidate.dim() == 2 and logp is None:
                logp = candidate
            elif candidate.dim() >= 3 and vector_field is None:
                vector_field = candidate

        if logp is None or vector_field is None:
            # Fall back: if both tensors share the same dimensionality (>2D),
            # reduce one copy to obtain row-wise log-probabilities.
            fallback = out1 if hasattr(out1, "dim") else out2
            if not hasattr(fallback, "dim") or fallback.dim() < 3:
                raise RuntimeError(
                    "Unexpected outputs from compute_posterior_x0_given_xt: "
                    f"shapes = {tuple(getattr(out1,'shape',()))}, {tuple(getattr(out2,'shape',()))}"
                )
            reduce_dims = tuple(range(2, fallback.dim()))
            logp = fallback.sum(dim=reduce_dims)
            logp = _torch.log_softmax(logp, dim=1)
            vector_field = fallback
        # Quick checks for uniformity when tau=1.0 (t_high == T)
        # Expected velocity: E[(x0 - eps_hat) | x_t] using posterior weights
        probs = logp.exp()  # (B, N)
        extra_dims = vector_field.dim() - probs.dim()
        if extra_dims > 0:
            probs_reshaped = probs.reshape(probs.shape + (1,) * extra_dims)
        else:
            probs_reshaped = probs
        expected_velocity = (probs_reshaped * vector_field).sum(dim=1)  # (B, C, H, W) or (B, D)

        # Ground-truth conditional velocity: x1 - x0 with x1 reconstructed from x_t_high
        sigma_h_safe = sigma_h.clamp_min(1e-8)
        eps_true = (x_t_high - mean_h * x0) / sigma_h_safe
        conditional_velocity = x0 - eps_true

        # Diagnostics comparing the two velocity estimates
        diff = expected_velocity - conditional_velocity
        exp_flat = expected_velocity.flatten(1)
        cond_flat = conditional_velocity.flatten(1)
        diff_flat = diff.flatten(1)

        exp_norm = exp_flat.norm(dim=1)
        cond_norm = cond_flat.norm(dim=1)
        dot = (exp_flat * cond_flat).sum(dim=1)
        denom = (exp_norm * cond_norm).clamp_min(1e-8)
        cosine = dot / denom
        mae = diff_flat.abs().mean(dim=1)
        rmse = diff_flat.pow(2).mean(dim=1).sqrt()
        norm_ratio = exp_norm / cond_norm.clamp_min(1e-8)

        print(
            "[stats] velocity expectation cosine (mean ± std): "
            f"{float(cosine.mean()):.4f} ± {float(cosine.std(unbiased=False)):.4f}"
        )
        print(
            "[stats] velocity MAE vs conditional (mean ± std): "
            f"{float(mae.mean()):.6f} ± {float(mae.std(unbiased=False)):.6f}"
        )
        print(
            "[stats] velocity RMSE vs conditional (mean ± std): "
            f"{float(rmse.mean()):.6f} ± {float(rmse.std(unbiased=False)):.6f}"
        )
        print(
            "[stats] velocity norm ratio (expected / conditional mean ± std): "
            f"{float(norm_ratio.mean()):.4f} ± {float(norm_ratio.std(unbiased=False)):.4f}"
        )

        if args.tau is not None and float(args.tau) == 1.0 and args.top_k is None:
            # Expect t_high == T exactly
            th = t_high.detach()
            assert torch.all(th == float(T)), (
                f"t_high not equal to T: unique={th.unique().tolist()} (T={T})"
            )
            # Alpha should be zero
            alpha = 1.0 - th / float(T)
            print(f"[debug] alpha min/max at tau=1.0: {float(alpha.min())} / {float(alpha.max())}")
            # Row-uniform probabilities expected (with uniform prior)
            probs_t = _torch.exp(logp)
            row_sum = probs_t.sum(dim=1)
            max_dev = float((probs_t - 1.0 / probs_t.size(1)).abs().max())
            print(f"[debug] row_sum min/max: {float(row_sum.min())} / {float(row_sum.max())}")
            print(f"[debug] max deviation from uniform: {max_dev}")
            # Soft assertions: warn rather than hard fail to help debugging
            if max_dev > 1e-5:
                print("[warn] tau=1.0 posterior not uniform; check prior/temperature/sigma_extra and selection of logp tensor.")
            assert _torch.allclose(row_sum, _torch.ones_like(row_sum), atol=1e-5), "Row sums not 1"
        mat = logp.exp().detach().cpu().numpy()

    if mat.ndim != 2:
        raise ValueError(f"Expected a 2D matrix, got shape {mat.shape}")

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 5), dpi=120)
    im = plt.imshow(mat, aspect="auto", cmap=args.cmap)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(args.title)
    plt.xlabel("Columns")
    plt.ylabel("Rows")
    plt.tight_layout()

    if args.save:
        plt.savefig(args.save)
    else:
        plt.show()


if __name__ == "__main__":
    main()
