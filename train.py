import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from configs import Config
from dataset import EEGPretrainDataset, collate_fn, build_mock_samples
from targets import RunningTargetNormalizer
from model import EEGPretrainModel
from losses import BalancedBandTrajectoryLoss
from trainer import train_one_epoch, validate_one_epoch
from utils import set_seed, ensure_dir_for_file


def main():
    cfg = Config()
    set_seed(cfg.train.seed)

    device = cfg.train.device if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # 每个 token 内部的目标时间帧数量 K
    frames_per_token = int(round(
        cfg.data.window_seconds / cfg.data.target_frame_seconds
    ))

    print(f"frames_per_token = {frames_per_token}")

    # 1. 构造 mock 数据
    samples = build_mock_samples(
        num_samples=64,
        clip_seconds=cfg.data.clip_seconds,
        sfreq=cfg.data.target_sfreq,
    )

    dataset = EEGPretrainDataset(samples, cfg)

    # 2. 划分训练集和验证集
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.train.seed),
    )

    # 3. 拟合 target normalizer
    n_bands = len(cfg.data.band_defs)

    normalizer = RunningTargetNormalizer(
        n_channels=cfg.data.n_channels,
        n_bands=n_bands,
        eps=cfg.data.eps,
    )

    print("Fitting target normalizer...")

    for i in train_set.indices:
        item = dataset[i]
        targets = item["targets"].numpy()  # [N, C, B, K]
        normalizer.fit_batch(targets)

    # target_mean: [C, B]
    # target_std:  [C, B]
    #
    # 变成可广播到:
    # targets: [B, N, C, BANDS, K]
    #
    # 所以 shape 为:
    # [1, 1, C, BANDS, 1]
    target_mean = torch.tensor(
        normalizer.mean,
        dtype=torch.float32,
        device=device,
    )[None, None, :, :, None]

    target_std = torch.tensor(
        np.sqrt(normalizer.var + cfg.data.eps),
        dtype=torch.float32,
        device=device,
    )[None, None, :, :, None]

    # 4. DataLoader
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

    # 5. 构建模型
    input_dim = cfg.data.n_channels * int(
        round(cfg.data.window_seconds * cfg.data.target_sfreq)
    )

    max_tokens = int(
        np.floor(
            (cfg.data.clip_seconds - cfg.data.window_seconds)
            / cfg.data.window_stride_seconds
        ) + 1
    )

    model = EEGPretrainModel(
        input_dim=input_dim,
        d_model=cfg.model.d_model,
        n_heads=cfg.model.n_heads,
        depth=cfg.model.depth,
        mlp_ratio=cfg.model.mlp_ratio,
        dropout=cfg.model.dropout,
        n_channels=cfg.data.n_channels,
        n_bands=n_bands,
        frames_per_token=frames_per_token,
        max_tokens=max_tokens,
    ).to(device)

    print(model)

    # 6. 优化器和 loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    criterion = BalancedBandTrajectoryLoss()

    # 7. 开始训练
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
                    "frames_per_token": frames_per_token,
                    "n_bands": n_bands,
                },
                cfg.train.save_path,
            )

            print(f"Saved best model to {cfg.train.save_path}")


if __name__ == "__main__":
    main()
