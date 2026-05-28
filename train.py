import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from configs import Config
from dataset import EEGPretrainDataset, collate_fn, build_mock_samples
from targets import RunningTargetNormalizer
from model import EEGPretrainModel
from losses import BalancedBandTokenTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from utils import set_seed, ensure_dir_for_file


def main():
    cfg = Config()
    set_seed(cfg.train.seed)

    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    patch_len = int(round(cfg.data.patch_seconds * cfg.data.target_sfreq))
    patch_stride = int(round(cfg.data.patch_stride_seconds * cfg.data.target_sfreq))
    total_points = int(round(cfg.data.clip_seconds * cfg.data.target_sfreq))

    num_time_patches = int(
        np.floor((total_points - patch_len) / patch_stride) + 1
    )

    num_tokens = cfg.data.n_channels * num_time_patches

    frames_per_patch = int(round(
        cfg.data.patch_seconds / cfg.data.target_frame_seconds
    ))

    n_bands = len(cfg.data.band_defs)

    print(f"patch_len = {patch_len}")
    print(f"patch_stride = {patch_stride}")
    print(f"total_points = {total_points}")
    print(f"num_time_patches = {num_time_patches}")
    print(f"num_tokens = {num_tokens}")
    print(f"frames_per_patch = {frames_per_patch}")
    print(f"n_bands = {n_bands}")

    samples = build_mock_samples(
        num_samples=64,
        clip_seconds=cfg.data.clip_seconds,
        sfreq=cfg.data.target_sfreq,
    )

    dataset = EEGPretrainDataset(samples, cfg)

    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    normalizer = RunningTargetNormalizer(
        n_channels=cfg.data.n_channels,
        n_bands=n_bands,
        eps=cfg.data.eps,
    )

    print("Fitting target normalizer...")

    for i in train_set.indices:
        item = dataset[i]

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
    )  # [C, F]

    target_std = torch.tensor(
        np.sqrt(normalizer.var + cfg.data.eps),
        dtype=torch.float32,
        device=device,
    )  # [C, F]

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_fn,
    )

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

    print(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTokenTrajectoryLoss()

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
                    "patch_len": patch_len,
                    "patch_stride": patch_stride,
                    "num_time_patches": num_time_patches,
                    "num_tokens": num_tokens,
                    "frames_per_patch": frames_per_patch,
                    "n_bands": n_bands,
                },
                cfg.train.save_path,
            )

            print(f"Saved best model to {cfg.train.save_path}")


if __name__ == "__main__":
    main()
