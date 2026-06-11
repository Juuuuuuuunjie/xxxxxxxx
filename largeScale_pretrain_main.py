import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt

from configs import Config
from channel_config import STANDARD_64_CHANNELS

from dataset import collate_fn
from lance_dataset import LanceEEGPretrainDataset
from prefetch_loader import PrefetchLoader

from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from masking import generate_block_mask, apply_token_mask
from utils import set_seed, ensure_dir_for_file


class BatchedSubset(Dataset):
    """
    保留底层 dataset.__getitems__ 批量读取能力的子集包装器。
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[int(idx)]]

    def __getitems__(self, idxs):
        mapped_indices = [self.indices[int(i)] for i in idxs]
        if hasattr(self.dataset, "__getitems__"):
            return self.dataset.__getitems__(mapped_indices)
        return [self.dataset[i] for i in mapped_indices]


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
    frames_per_patch = int(round(cfg.data.patch_seconds / cfg.data.target_frame_seconds))
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
        )
        masks.append(mask)

    return torch.tensor(
        np.stack(masks, axis=0),
        dtype=torch.float32,
        device=device,
    )


def move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.device.type == "cpu":
                moved[key] = value.to(device, non_blocking=True)
            else:
                moved[key] = value
        else:
            moved[key] = value
    return moved


@torch.no_grad()
def visualize_train_reconstruction(
    model,
    loader,
    cfg,
    device,
    save_dir,
    max_samples=4,
):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)

    token_inputs = batch["token_inputs"]
    targets = batch["targets"]
    token_channel_indices = batch["token_channel_indices"]
    token_time_indices = batch["token_time_indices"]
    token_valid_mask = batch["token_valid_mask"]

    B, S, _ = token_inputs.shape

    token_mask = build_batch_token_mask(B, S, cfg, device)
    masked_inputs = apply_token_mask(tokens=token_inputs, token_mask=token_mask)

    pred_norm = model(
        token_inputs=masked_inputs,
        token_channel_indices=token_channel_indices,
        token_time_indices=token_time_indices,
    )

    pred = pred_norm.detach().cpu().numpy()
    target = targets.detach().cpu().numpy()
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

        masked_token_idx = int(np.random.choice(masked_valid_indices))
        unmasked_token_idx = int(np.random.choice(unmasked_valid_indices))

        examples = [("masked", masked_token_idx), ("unmasked", unmasked_token_idx)]

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
                ax.plot(x, target[b, tok_idx, f], marker="o", linewidth=2, label="target")
                ax.plot(x, pred[b, tok_idx, f], marker="x", linewidth=2, label="pred")
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

    device = cfg.train.device
    print("Using device:", device)

    shape_params = compute_shape_params(cfg)

    patch_len = shape_params["patch_len"]
    num_time_patches = shape_params["num_time_patches"]
    frames_per_patch = shape_params["frames_per_patch"]
    n_bands = shape_params["n_bands"]

    print("\n===== Shape Params =====")
    for k, v in shape_params.items():
        print(f"{k}: {v}")

    print("\n===== Loading Processed EEG Samples From Lance =====")

    processed_lance_path = "./serverData/eeg_openneuro_76ch_val_processed.lance"

    dataset = LanceEEGPretrainDataset(
        lance_path=processed_lance_path,
    )

    print("processed_lance_path:", processed_lance_path)
    print("dataset size:", len(dataset))

    if len(dataset) < 2:
        raise ValueError(
            "Too few samples. Need at least 2 for train/val split. "
            "Please regenerate Lance with shorter clip_seconds or use longer recordings."
        )

    s0 = dataset[0]

    print("\nFirst Lance sample:")
    print("  token_inputs:          ", tuple(s0["token_inputs"].shape))
    print("  targets:               ", tuple(s0["targets"].shape))
    print("  token_channel_indices: ", tuple(s0["token_channel_indices"].shape))
    print("  token_time_indices:    ", tuple(s0["token_time_indices"].shape))
    print("  token_valid_mask:      ", tuple(s0["token_valid_mask"].shape))
    print("  channel_valid_mask:    ", tuple(s0["channel_valid_mask"].shape))
    print("  valid channels:        ", int(s0["channel_valid_mask"].sum().item()), "/ 64")

    print("dataset size:", len(dataset))

    n_total = len(dataset)
    n_train = max(1, int(n_total * 0.8))
    n_val = n_total - n_train

    if n_val == 0:
        n_train = n_total - 1
        n_val = 1

    split_generator = torch.Generator().manual_seed(cfg.train.seed)
    all_indices = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = all_indices[:n_train]
    val_indices = all_indices[n_train:]

    train_set = BatchedSubset(dataset, train_indices)
    val_set = BatchedSubset(dataset, val_indices)

    print(f"\n===== Dataset Split =====")
    print(f"train: {len(train_set)}, val: {len(val_set)}")

    train_batch_size = min(cfg.train.batch_size, len(train_set))
    val_batch_size = min(cfg.train.batch_size, len(val_set))

    base_train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    base_val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=False,
    )

    use_prefetch_loader = getattr(cfg.train, "use_prefetch_loader", True)
    prefetch = getattr(cfg.train, "prefetch", 2)

    if use_prefetch_loader:
        train_loader = PrefetchLoader(
            base_train_loader,
            device=device,
            prefetch=prefetch,
        )
        val_loader = PrefetchLoader(
            base_val_loader,
            device=device,
            prefetch=prefetch,
        )
        print(f"PrefetchLoader enabled: prefetch={prefetch}")
    else:
        train_loader = base_train_loader
        val_loader = base_val_loader
        print("PrefetchLoader disabled")

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
        )

        val_loss = validate_one_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            cfg=cfg,
        )

        print(f"Epoch {epoch + 1}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            ensure_dir_for_file(cfg.train.save_path)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "shape_params": shape_params,
                },
                cfg.train.save_path,
            )
            print(f"Saved best model to {cfg.train.save_path}")

    print("\n===== Visualizing Train Reconstruction =====")
    visualize_train_reconstruction(
        model=model,
        loader=train_loader,
        cfg=cfg,
        device=device,
        save_dir=cfg.train.vis_dir,
        max_samples=4,
    )

    print("\nDone.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Visualization saved to: {cfg.train.vis_dir}")


if __name__ == "__main__":
    main()
