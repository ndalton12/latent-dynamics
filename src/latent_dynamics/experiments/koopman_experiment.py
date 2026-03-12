from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from latent_dynamics.activations import extract_hidden_trajectories
from latent_dynamics.config import MODEL_REGISTRY, RunConfig
from latent_dynamics.data import load_examples, prepare_text_and_labels
from latent_dynamics.hub import load_activations
from latent_dynamics.models import load_model_and_tokenizer, resolve_device


@dataclass
class ExperimentConfig:
    activations: Path | None = None
    model_key: str = "gemma3_4b"
    dataset_key: str = "toy_contrastive"
    split: str = "train"
    layer_idx: int = 5
    max_samples: int = 120
    max_input_tokens: int = 256
    use_generate: bool = False
    max_new_tokens: int = 24
    include_prompt_in_trajectory: bool = True
    device: str | None = None
    latent_dim: int = 128
    hidden_dim: int = 512
    train_fraction: float = 0.8
    batch_size: int = 1024
    epochs: int = 80
    lr: float = 1e-3
    weight_decay: float = 1e-5
    recon_weight: float = 1.0
    pred_weight: float = 1.0
    latent_weight: float = 1.0
    k_l2_weight: float = 1e-4
    enforce_spectral_norm: bool = False
    spectral_norm_target: float = 1.0
    grad_clip: float = 1.0
    patience: int = 12
    random_state: int = 7
    log_every: int = 10
    output_json: Path | None = None


class KoopmanAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.K = nn.Parameter(torch.eye(latent_dim))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def step_latent(self, z_t: torch.Tensor) -> torch.Tensor:
        return z_t @ self.K.T

    def forward_pair(
        self,
        x_t: torch.Tensor,
        x_tp1: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_t = self.encode(x_t)
        z_tp1 = self.encode(x_tp1)
        z_pred = self.step_latent(z_t)
        x_t_hat = self.decode(z_t)
        x_tp1_hat = self.decode(z_tp1)
        x_tp1_pred = self.decode(z_pred)
        return {
            "z_t": z_t,
            "z_tp1": z_tp1,
            "z_pred": z_pred,
            "x_t_hat": x_t_hat,
            "x_tp1_hat": x_tp1_hat,
            "x_tp1_pred": x_tp1_pred,
        }


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _split_indices(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if n < 2:
        raise ValueError(
            "Need at least 2 trajectories to form train/validation splits."
        )
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(frac * n))
    n_train = min(max(1, n_train), n - 1)
    return perm[:n_train], perm[n_train:]


def _collect_states(trajectories: list[np.ndarray], idxs: np.ndarray) -> np.ndarray:
    return np.concatenate([trajectories[i] for i in idxs], axis=0).astype(np.float32)


def _collect_pairs(
    trajectories: list[np.ndarray],
    idxs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in idxs:
        traj = trajectories[int(i)].astype(np.float32)
        if traj.shape[0] < 2:
            continue
        xs.append(traj[:-1])
        ys.append(traj[1:])
    if not xs:
        raise ValueError(
            "No trajectory pairs available. Increase max_samples or sequence length."
        )
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def _prepare_trajectories(cfg: ExperimentConfig) -> tuple[list[np.ndarray], RunConfig]:
    if cfg.activations is not None:
        trajectories, _, _, _, _, stored_cfg = load_activations(cfg.activations)
        return trajectories, stored_cfg

    run_cfg = RunConfig(
        model_key=cfg.model_key,
        dataset_key=cfg.dataset_key,
        max_samples=cfg.max_samples,
        max_input_tokens=cfg.max_input_tokens,
        layer_idx=cfg.layer_idx,
        device=resolve_device(cfg.device),
        use_generate=cfg.use_generate,
        max_new_tokens=cfg.max_new_tokens,
        include_prompt_in_trajectory=cfg.include_prompt_in_trajectory,
    )

    ds, spec = load_examples(run_cfg.dataset_key, run_cfg.max_samples)
    texts, _labels = prepare_text_and_labels(
        ds,
        text_field=spec.text_field,
        label_field=spec.label_field,
        label_fn=spec.label_fn,
    )
    model, tokenizer = load_model_and_tokenizer(
        run_cfg.model_key, run_cfg.device or "cpu"
    )
    result = extract_hidden_trajectories(
        model=model, tokenizer=tokenizer, texts=texts,
        layer_idx=run_cfg.layer_idx, cfg=run_cfg,
    )
    trajectories = result.per_layer[run_cfg.layer_idx]
    return trajectories, run_cfg


def _denormalize(
    x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    return (x * std) + mean


def _compute_pair_metrics(
    model: KoopmanAutoencoder,
    x_t: np.ndarray,
    x_tp1: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    batch_size: int,
    device: str,
) -> dict[str, float]:
    model.eval()
    ds = TensorDataset(
        torch.from_numpy(((x_t - mean) / std).astype(np.float32)),
        torch.from_numpy(((x_tp1 - mean) / std).astype(np.float32)),
        torch.from_numpy(x_tp1.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    total = 0
    latent_mse_sum = 0.0
    state_norm_mse_sum = 0.0
    state_raw_mse_sum = 0.0

    with torch.no_grad():
        for x_t_b, x_tp1_b, x_tp1_raw_b in dl:
            x_t_b = x_t_b.to(device)
            x_tp1_b = x_tp1_b.to(device)
            x_tp1_raw_b = x_tp1_raw_b.to(device)

            out = model.forward_pair(x_t_b, x_tp1_b)
            latent_mse = torch.mean((out["z_pred"] - out["z_tp1"]) ** 2, dim=1)
            state_norm_mse = torch.mean((out["x_tp1_pred"] - x_tp1_b) ** 2, dim=1)
            pred_raw = _denormalize(out["x_tp1_pred"], mean_t, std_t)
            state_raw_mse = torch.mean((pred_raw - x_tp1_raw_b) ** 2, dim=1)

            bs = x_t_b.shape[0]
            total += bs
            latent_mse_sum += float(latent_mse.sum().item())
            state_norm_mse_sum += float(state_norm_mse.sum().item())
            state_raw_mse_sum += float(state_raw_mse.sum().item())

    return {
        "latent_mse": latent_mse_sum / max(total, 1),
        "state_mse_normalized": state_norm_mse_sum / max(total, 1),
        "state_mse_raw": state_raw_mse_sum / max(total, 1),
    }


def _rollout_metrics(
    model: KoopmanAutoencoder,
    trajectories: list[np.ndarray],
    idxs: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
    horizons: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    model.eval()
    mean_t = torch.from_numpy(mean).to(device)
    std_t = torch.from_numpy(std).to(device)

    total_steps = 0
    raw_err_sum = 0.0
    latent_err_sum = 0.0
    horizon_sum: dict[int, float] = {h: 0.0 for h in horizons}
    horizon_count: dict[int, int] = {h: 0 for h in horizons}

    with torch.no_grad():
        for idx in idxs:
            traj = trajectories[int(idx)].astype(np.float32)
            if traj.shape[0] < 2:
                continue

            x_norm = (traj - mean) / std
            x_norm_t = torch.from_numpy(x_norm).to(device)
            x_raw_t = torch.from_numpy(traj).to(device)

            z = model.encode(x_norm_t[:1])[0]
            for step in range(1, traj.shape[0]):
                z = model.step_latent(z)
                x_pred_norm = model.decode(z)
                x_true_norm = x_norm_t[step]
                x_true_raw = x_raw_t[step]

                z_true = model.encode(x_true_norm.unsqueeze(0))[0]
                latent_err = torch.mean((z - z_true) ** 2)
                x_pred_raw = _denormalize(x_pred_norm, mean_t, std_t)
                raw_err = torch.mean((x_pred_raw - x_true_raw) ** 2)

                raw_err_sum += float(raw_err.item())
                latent_err_sum += float(latent_err.item())
                total_steps += 1

                if step in horizon_sum:
                    horizon_sum[step] += float(raw_err.item())
                    horizon_count[step] += 1

    metrics: dict[str, float] = {
        "rollout_state_mse_raw": raw_err_sum / max(total_steps, 1),
        "rollout_latent_mse": latent_err_sum / max(total_steps, 1),
        "rollout_steps": float(total_steps),
    }
    for h in horizons:
        if horizon_count[h] > 0:
            metrics[f"rollout_state_mse_raw_h{h}"] = horizon_sum[h] / horizon_count[h]
    return metrics


def _fit_pca_baseline(
    train_states: np.ndarray,
    x_t_train: np.ndarray,
    x_tp1_train: np.ndarray,
    x_t_val: np.ndarray,
    x_tp1_val: np.ndarray,
    latent_dim: int,
    seed: int,
) -> dict[str, float]:
    max_rank = int(min(train_states.shape[0], train_states.shape[1]))
    if max_rank < 1:
        raise ValueError("PCA baseline requires at least one training state.")

    effective_dim = int(min(latent_dim, max_rank))
    if effective_dim != latent_dim:
        print(
            f"[pca baseline] requested latent_dim={latent_dim} exceeds max_rank={max_rank}; "
            f"using n_components={effective_dim}."
        )

    pca = PCA(n_components=effective_dim, random_state=seed)
    pca.fit(train_states)

    z_t_train = pca.transform(x_t_train)
    z_tp1_train = pca.transform(x_tp1_train)
    z_t_val = pca.transform(x_t_val)
    z_tp1_val = pca.transform(x_tp1_val)

    k_t, *_ = np.linalg.lstsq(z_t_train, z_tp1_train, rcond=None)
    z_pred_val = z_t_val @ k_t
    x_pred_val = pca.inverse_transform(z_pred_val)

    latent_mse = float(np.mean((z_pred_val - z_tp1_val) ** 2))
    state_mse = float(np.mean((x_pred_val - x_tp1_val) ** 2))
    return {
        "pca_koopman_effective_dim": float(effective_dim),
        "pca_koopman_latent_mse": latent_mse,
        "pca_koopman_state_mse_raw": state_mse,
    }


def _parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser(
        description="Deep Koopman autoencoder experiment on latent trajectories.",
    )
    parser.add_argument("--activations", type=Path, default=None)
    parser.add_argument(
        "--model-key", choices=sorted(MODEL_REGISTRY.keys()), default="gemma3_4b"
    )
    parser.add_argument("--dataset-key", default="toy_contrastive")
    parser.add_argument("--split", default="train")
    parser.add_argument("--layer-idx", type=int, default=5)
    parser.add_argument("--max-samples", type=int, default=120)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--use-generate", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=36)
    parser.add_argument("--no-include-prompt", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--recon-weight", type=float, default=1.0)
    parser.add_argument("--pred-weight", type=float, default=1.0)
    parser.add_argument("--latent-weight", type=float, default=1.0)
    parser.add_argument("--k-l2-weight", type=float, default=1e-4)
    parser.add_argument("--enforce-spectral-norm", action="store_true")
    parser.add_argument("--spectral-norm-target", type=float, default=1.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    return ExperimentConfig(
        activations=args.activations,
        model_key=args.model_key,
        dataset_key=args.dataset_key,
        split=args.split,
        layer_idx=args.layer_idx,
        max_samples=args.max_samples,
        max_input_tokens=args.max_input_tokens,
        use_generate=args.use_generate,
        max_new_tokens=args.max_new_tokens,
        include_prompt_in_trajectory=(not args.no_include_prompt),
        device=args.device,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        train_fraction=args.train_fraction,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        recon_weight=args.recon_weight,
        pred_weight=args.pred_weight,
        latent_weight=args.latent_weight,
        k_l2_weight=args.k_l2_weight,
        enforce_spectral_norm=args.enforce_spectral_norm,
        spectral_norm_target=args.spectral_norm_target,
        grad_clip=args.grad_clip,
        patience=args.patience,
        random_state=args.random_state,
        log_every=args.log_every,
        output_json=args.output_json,
    )


def _project_spectral_norm_(matrix: torch.Tensor, target: float) -> float:
    """Project matrix onto spectral-norm ball: ||matrix||_2 <= target."""
    if target <= 0:
        raise ValueError("spectral_norm_target must be > 0.")
    with torch.no_grad():
        sigma_val = float(torch.linalg.svdvals(matrix.detach().float().cpu())[0].item())
        if np.isfinite(sigma_val) and sigma_val > target:
            matrix.mul_(target / (sigma_val + 1e-12))
    return sigma_val


def run_experiment(cfg: ExperimentConfig) -> dict[str, Any]:
    set_seed(cfg.random_state)

    trajectories, source_cfg = _prepare_trajectories(cfg)
    if not trajectories:
        raise ValueError("No trajectories found.")

    input_dim = int(trajectories[0].shape[-1])
    if cfg.latent_dim >= input_dim:
        raise ValueError(
            f"latent_dim must be smaller than input_dim. Got latent_dim={cfg.latent_dim}, input_dim={input_dim}."
        )

    train_idxs, val_idxs = _split_indices(
        len(trajectories), cfg.train_fraction, cfg.random_state
    )
    train_states = _collect_states(trajectories, train_idxs)
    mean = train_states.mean(axis=0).astype(np.float32)
    std = (train_states.std(axis=0) + 1e-6).astype(np.float32)

    x_t_train, x_tp1_train = _collect_pairs(trajectories, train_idxs)
    x_t_val, x_tp1_val = _collect_pairs(trajectories, val_idxs)

    x_t_train_n = ((x_t_train - mean) / std).astype(np.float32)
    x_tp1_train_n = ((x_tp1_train - mean) / std).astype(np.float32)

    device = resolve_device(cfg.device)
    model = KoopmanAutoencoder(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_t_train_n),
            torch.from_numpy(x_tp1_train_n),
        ),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_val = float("inf")
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    stale_epochs = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for x_t_b, x_tp1_b in train_loader:
            x_t_b = x_t_b.to(device)
            x_tp1_b = x_tp1_b.to(device)

            out = model.forward_pair(x_t_b, x_tp1_b)
            recon_loss = torch.mean((out["x_t_hat"] - x_t_b) ** 2) + torch.mean(
                (out["x_tp1_hat"] - x_tp1_b) ** 2
            )
            pred_loss = torch.mean((out["x_tp1_pred"] - x_tp1_b) ** 2)
            latent_loss = torch.mean((out["z_pred"] - out["z_tp1"]) ** 2)
            k_l2 = torch.mean(model.K**2)

            loss = (
                cfg.recon_weight * recon_loss
                + cfg.pred_weight * pred_loss
                + cfg.latent_weight * latent_loss
                + cfg.k_l2_weight * k_l2
            )

            optimizer.zero_grad()
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if cfg.enforce_spectral_norm:
                _project_spectral_norm_(model.K, cfg.spectral_norm_target)

            epoch_loss += float(loss.item())
            batch_count += 1

        train_loss = epoch_loss / max(batch_count, 1)
        val_metrics = _compute_pair_metrics(
            model=model,
            x_t=x_t_val,
            x_tp1=x_tp1_val,
            mean=mean,
            std=std,
            batch_size=cfg.batch_size,
            device=device,
        )
        val_loss = val_metrics["state_mse_normalized"] + val_metrics["latent_mse"]

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        entry = {
            "epoch": float(epoch),
            "train_loss": train_loss,
            "val_latent_mse": val_metrics["latent_mse"],
            "val_state_mse_raw": val_metrics["state_mse_raw"],
            "val_state_mse_normalized": val_metrics["state_mse_normalized"],
        }
        history.append(entry)

        if cfg.log_every > 0 and (epoch == 1 or epoch % cfg.log_every == 0):
            print(
                f"[epoch {epoch:03d}] "
                f"train={train_loss:.5f} "
                f"val_lat={val_metrics['latent_mse']:.5f} "
                f"val_state={val_metrics['state_mse_raw']:.5f}"
            )

        if stale_epochs >= cfg.patience:
            print(f"Early stopping at epoch {epoch} (patience={cfg.patience}).")
            break

    model.load_state_dict(best_state)
    model.eval()

    pair_metrics = _compute_pair_metrics(
        model=model,
        x_t=x_t_val,
        x_tp1=x_tp1_val,
        mean=mean,
        std=std,
        batch_size=cfg.batch_size,
        device=device,
    )
    rollout = _rollout_metrics(
        model=model,
        trajectories=trajectories,
        idxs=val_idxs,
        mean=mean,
        std=std,
        device=device,
    )
    baseline = _fit_pca_baseline(
        train_states=train_states,
        x_t_train=x_t_train,
        x_tp1_train=x_tp1_train,
        x_t_val=x_t_val,
        x_tp1_val=x_tp1_val,
        latent_dim=cfg.latent_dim,
        seed=cfg.random_state,
    )

    eigvals = torch.linalg.eigvals(model.K.detach().cpu())
    spectral_radius = float(torch.max(torch.abs(eigvals)).item())

    results: dict[str, Any] = {
        "experiment": asdict(cfg),
        "source_config": asdict(source_cfg),
        "data": {
            "n_trajectories": len(trajectories),
            "train_trajectories": int(len(train_idxs)),
            "val_trajectories": int(len(val_idxs)),
            "train_pairs": int(len(x_t_train)),
            "val_pairs": int(len(x_t_val)),
            "input_dim": input_dim,
            "latent_dim": cfg.latent_dim,
        },
        "metrics": {
            **pair_metrics,
            **rollout,
            **baseline,
            "koopman_spectral_radius": spectral_radius,
            "best_val_objective": best_val,
        },
        "training_history": history,
    }
    return results


def main() -> None:
    cfg = _parse_args()
    results = run_experiment(cfg)
    report = json.dumps(results, indent=2, default=str)
    print(report)

    if cfg.output_json is not None:
        cfg.output_json.parent.mkdir(parents=True, exist_ok=True)
        cfg.output_json.write_text(report)
        print(f"Wrote results: {cfg.output_json}")


if __name__ == "__main__":
    main()
