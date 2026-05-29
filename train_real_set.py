import os
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

import matplotlib.pyplot as plt

from configs import Config
from channel_config import STANDARD_64_CHANNELS
from io_utils import load_multiple_eeglab_sets_as_samples
from dataset import EEGPretrainDataset, collate_fn
from targets import RunningTargetNormalizer
from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from masking import generate_block_mask, apply_token_mask
from utils import set_seed, ensure_dir_for_file


def compute_shape_params(cfg):
    patch_len = int(round(cfg.data.patch_seconds * cfg.data.target_sfreq))
    patch_stride = int(round(cfg.data.patch_stride_seconds * cfg.data.target_sfreq))
    total_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))

    num_time_patches = int(
        np.floor((total_points - patch_len) / patch_stride) + 1
    )

    if num_time_patches <= 0:
        raise ValueError(
            f"num_time_patches <= 0. "
            f"Please check clip_seconds={cfg.data.clip_seconds}, "
            f"patch_seconds={cfg.data.patch_seconds}, "
            f"target_sfreq={cfg.data.target_sfreq}"
        )

    num_tokens = cfg.data.n_channels * num_time_patches

    frames_per_patch = int(round(
        cfg.data.patch_seconds / cfg.data.target_frame_seconds
    ))

    n_bands = len(cfg.data.band_defs)

    return {
        "patch_len": patch_len,
        "patch_stride": patch_stride,
        "total_points": total_points,
        "num_time_patches": num_time_patches,
        "num_tokens": num_tokens,
        "frames_per_patch": frames_per_patch,
        "n_bands": n_bands,
    }


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = []

    for _ in range(batch_size):
        mask = generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
            min_block_tokens=cfg.mask.min_block_tokens,
            max_block_tokens=cfg.mask.max_block_tokens,
        )
        masks.append(mask)

    token_mask = torch.tensor(
        np.stack(masks, axis=0),
        dtype=torch.float32,
        device=device,
    )

    return token_mask


def gather_token_normalizer(
    token_channel_indices,
    target_mean,
    target_std,
):
    """
    Args:
        token_channel_indices: [B, S]
        target_mean: [C, F]
        target_std: [C, F]

    Returns:
        token_mean: [B, S, F, 1]
        token_std: [B, S, F, 1]
    """
    token_mean = target_mean[token_channel_indices]  # [B, S, F]
    token_std = target_std[token_channel_indices]    # [B, S, F]

    token_mean = token_mean[:, :, :, None]
    token_std = token_std[:, :, :, None]

    return token_mean, token_std


@torch.no_grad()
def visualize_train_reconstruction(
    model,
    loader,
    cfg,
    device,
    target_mean,
    target_std,
    save_dir,
    max_samples=4,
):
    """
    在训练集上可视化 masked token 和 unmasked token 的重构效果。

    注意：
    当前模型重构的是时频目标，不是原始波形。
    每张图会展示：
        - 一个 masked token
        - 一个 unmasked token
    每个 token 内画多个频段的 target vs pred 曲线。
    """
    os.makedirs(save_dir, exist_ok=True)

    model.eval()

    batch = next(iter(loader))

    token_inputs = batch["token_inputs"].to(device)                    # [B, S, L]
    targets = batch["targets"].to(device)                              # [B, S, F, K]
    token_channel_indices = batch["token_channel_indices"].to(device)  # [B, S]
    token_time_indices = batch["token_time_indices"].to(device)        # [B, S]
    token_valid_mask = batch["token_valid_mask"].to(device)            # [B, S]

    B, S, L = token_inputs.shape

    token_mask = build_batch_token_mask(
        batch_size=B,
        num_tokens=S,
        cfg=cfg,
        device=device,
    )

    masked_inputs = apply_token_mask(
        tokens=token_inputs,
        token_mask=token_mask,
    )

    token_mean, token_std = gather_token_normalizer(
        token_channel_indices=token_channel_indices,
        target_mean=target_mean,
        target_std=target_std,
    )

    normalized_targets = (targets - token_mean) / token_std

    pred_norm = model(
        token_inputs=masked_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
    )

    # 反标准化，方便看真实尺度下的效果
    pred = pred_norm * token_std + token_mean
    target = targets

    pred = pred.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    token_mask_np = token_mask.detach().cpu().numpy()
    token_valid_np = token_valid_mask.detach().cpu().numpy()
    token_ch_np = token_channel_indices.detach().cpu().numpy()
    token_t_np = token_time_indices.detach().cpu().numpy()

    band_names = [x[0] for x in cfg.data.band_defs]

    num_to_plot = min(max_samples, B)

    for b in range(num_to_plot):
        valid_indices = np.where(token_valid_np[b] > 0.5)[0]
        masked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] > 0.5)
        )[0]
        unmasked_valid_indices = np.where(
            (token_valid_np[b] > 0.5) & (token_mask_np[b] < 0.5)
        )[0]

        if len(valid_indices) == 0:
            print(f"[Visualization] sample {b}: no valid token, skip.")
            continue

        if len(masked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no masked valid token, skip.")
            continue

        if len(unmasked_valid_indices) == 0:
            print(f"[Visualization] sample {b}: no unmasked valid token, skip.")
            continue

        masked_token_idx = int(masked_valid_indices[0])
        unmasked_token_idx = int(unmasked_valid_indices[0])

        examples = [
            ("masked", masked_token_idx),
            ("unmasked", unmasked_token_idx),
        ]

        F = target.shape[2]
        K = target.shape[3]
        x = np.arange(K)

        fig, axes = plt.subplots(
            nrows=F,
            ncols=2,
            figsize=(12, 2.2 * F),
            squeeze=False,
        )

        for col, (case_name, tok_idx) in enumerate(examples):
            ch_idx = int(token_ch_np[b, tok_idx])
            time_idx = int(token_t_np[b, tok_idx])
            ch_name = STANDARD_64_CHANNELS[ch_idx]

            for f in range(F):
                ax = axes[f][col]

                y_true = target[b, tok_idx, f]
                y_pred = pred[b, tok_idx, f]

                ax.plot(x, y_true, marker="o", linewidth=2, label="target")
                ax.plot(x, y_pred, marker="x", linewidth=2, label="pred")

                ax.set_title(
                    f"{case_name} | token={tok_idx} | {ch_name} | patch={time_idx} | {band_names[f]}"
                )
                ax.set_xlabel("frame inside patch")
                ax.set_ylabel("log band power")
                ax.grid(True, alpha=0.3)

                if f == 0:
                    ax.legend()

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"train_reconstruction_sample_{b}.png")
        plt.savefig(save_path, dpi=150)
        plt.close(fig)

        print(f"[Visualization] saved: {save_path}")


def main():
    cfg = Config()
    set_seed(cfg.train.seed)

    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    shape_params = compute_shape_params(cfg)

    patch_len = shape_params["patch_len"]
    num_time_patches = shape_params["num_time_patches"]
    num_tokens = shape_params["num_tokens"]
    frames_per_patch = shape_params["frames_per_patch"]
    n_bands = shape_params["n_bands"]

    print("\n===== Shape Params =====")
    for k, v in shape_params.items():
        print(f"{k}: {v}")

    print("\n===== Standard 64 Channels =====")
    print(STANDARD_64_CHANNELS)

    # 1) 读取 .set 数据，并切成固定长度样本
    print("\n===== Loading Real EEG .set Files =====")

    samples = load_multiple_eeglab_sets_as_samples(
        set_paths=cfg.data.set_paths,
        clip_seconds=cfg.data.clip_seconds,
        clip_stride_seconds=cfg.data.clip_stride_seconds,
        convert_v_to_uv=cfg.data.convert_v_to_uv,
    )

    print(f"\nTotal clips: {len(samples)}")

    if len(samples) < 2:
        raise ValueError(
            "Too few clips. Need at least 2 clips for train/val split. "
            "You can reduce cfg.data.clip_seconds or use longer EEG recording."
        )

    first = samples[0]
    print("\nFirst raw clip:")
    print("signal shape:", first["signal"].shape)
    print("sfreq:", first["sfreq"])
    print("first 20 channel names:", first["channel_names"][:20])

    # 2) 构建 Dataset
    dataset = EEGPretrainDataset(samples, cfg)

    # 3) train/val split
    n_total = len(dataset)
    n_train = max(1, int(n_total * 0.8))
    n_val = n_total - n_train

    if n_val == 0:
        n_train = n_total - 1
        n_val = 1

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    print("\n===== Dataset Split =====")
    print("train samples:", len(train_set))
    print("val samples:", len(val_set))

    # 4) 先取一个样本检查 shape
    print("\n===== Checking One Processed Sample =====")
    item0 = dataset[0]
    print("token_inputs:", item0["token_inputs"].shape)
    print("targets:", item0["targets"].shape)
    print("token_channel_indices:", item0["token_channel_indices"].shape)
    print("token_time_indices:", item0["token_time_indices"].shape)
    print("token_valid_mask:", item0["token_valid_mask"].shape)
    print("channel_valid_mask:", item0["channel_valid_mask"].shape)
    print("valid channels:", int(item0["channel_valid_mask"].sum().item()), "/ 64")

    # 5) 拟合 target normalizer
    print("\n===== Fitting Target Normalizer =====")

    normalizer = RunningTargetNormalizer(
        n_channels=cfg.data.n_channels,
        n_bands=n_bands,
        eps=cfg.data.eps,
    )

    for idx_in_subset in train_set.indices:
        item = dataset[idx_in_subset]

        targets_flat = item["targets"].numpy()  # [S, F, K]

        C = cfg.data.n_channels
        N = num_time_patches
        F = n_bands
        K = frames_per_patch

        targets = targets_flat.reshape(C, N, F, K)

        normalizer.fit_batch(targets)

    target_mean = torch.tensor(
        normalizer.mean,
        dtype=torch.float32,
        device=device,
    )

    target_std = torch.tensor(
        np.sqrt(normalizer.var + cfg.data.eps),
        dtype=torch.float32,
        device=device,
    )

    # 6) DataLoader
    train_loader = DataLoader(
        train_set,
        batch_size=min(cfg.train.batch_size, len(train_set)),
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=min(cfg.train.batch_size, len(val_set)),
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    # 7) 模型
    print("\n===== Building Model =====")

    model = EEGPretrainModel(
        input_dim=patch_len,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        depth=cfg.model.depth,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        n_channels=cfg.data.n_channels,
        n_time_patches=num_time_patches,
        n_bands=n_bands,
        frames_per_patch=frames_per_patch,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTokenTrajectoryLoss()

    # 8) 训练
    print("\n===== Start Training =====")

    best_val = float("inf")

    for epoch in range(cfg.train.num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{cfg.train.num_epochs} =====")

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
            target_mean=target_mean,
            target_std=target_std,
        )

        print(
            f"Epoch {epoch + 1}: "
            f"train_loss={train_loss:.6f}, "
            f"val_loss={val_loss:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss

            ensure_dir_for_file(cfg.train.save_path)

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "target_mean": target_mean.cpu(),
                    "target_std": target_std.cpu(),
                    "config": cfg,
                    "shape_params": shape_params,
                },
                cfg.train.save_path,
            )

            print(f"Saved best model to {cfg.train.save_path}")

    # 9) 训练结束后，在训练集上可视化重构效果
    print("\n===== Visualizing Train Reconstruction =====")

    visualize_train_reconstruction(
        model=model,
        loader=train_loader,
        cfg=cfg,
        device=device,
        target_mean=target_mean,
        target_std=target_std,
        save_dir=cfg.train.vis_dir,
        max_samples=4,
    )

    print("\nDone.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Visualization saved to: {cfg.train.vis_dir}")


if __name__ == "__main__":
    main()
