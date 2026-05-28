import numpy as np
import torch

from masking import generate_block_mask, apply_token_mask


def fit_target_normalizer(dataset, normalizer):
    """
    用训练集统计 target 的 mean/std。
    """
    for i in range(len(dataset)):
        item = dataset[i]
        targets = item["targets"].numpy()  # [N, C, B, K]
        normalizer.fit_batch(targets)


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    """
    为一个 batch 生成时间 token mask。

    mask 只用于破坏输入，不限制 loss 计算范围。
    """
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


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg,
    target_mean,
    target_std,
):
    model.train()

    total_loss = 0.0

    for step, batch in enumerate(loader):
        token_inputs = batch["token_inputs"].to(device)              # [B, N, D]
        targets = batch["targets"].to(device)                        # [B, N, C, F, K]
        channel_valid_mask = batch["channel_valid_mask"].to(device)  # [B, C]

        batch_size, num_tokens, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=batch_size,
            num_tokens=num_tokens,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        # target_mean / target_std:
        # [1, 1, C, F, 1]
        normalized_targets = (targets - target_mean) / target_std

        pred = model(masked_inputs)  # [B, N, C, F, K]

        loss = criterion(
            pred=pred,
            target=normalized_targets,
            channel_valid_mask=channel_valid_mask,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if step % cfg.train.print_every == 0:
            print(
                f"step={step}, "
                f"loss={loss.item():.6f}, "
                f"input_mask_ratio={token_mask.mean().item():.3f}"
            )

    return total_loss / max(1, len(loader))


@torch.no_grad()
def validate_one_epoch(
    model,
    loader,
    criterion,
    device,
    cfg,
    target_mean,
    target_std,
):
    model.eval()

    total_loss = 0.0

    for batch in loader:
        token_inputs = batch["token_inputs"].to(device)              # [B, N, D]
        targets = batch["targets"].to(device)                        # [B, N, C, F, K]
        channel_valid_mask = batch["channel_valid_mask"].to(device)  # [B, C]

        batch_size, num_tokens, _ = token_inputs.shape

        token_mask = build_batch_token_mask(
            batch_size=batch_size,
            num_tokens=num_tokens,
            cfg=cfg,
            device=device,
        )

        masked_inputs = apply_token_mask(
            tokens=token_inputs,
            token_mask=token_mask,
        )

        normalized_targets = (targets - target_mean) / target_std

        pred = model(masked_inputs)

        loss = criterion(
            pred=pred,
            target=normalized_targets,
            channel_valid_mask=channel_valid_mask,
        )

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
