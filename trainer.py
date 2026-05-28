import numpy as np
import torch

from masking import generate_block_mask, apply_token_mask


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
    token_mean = target_mean[token_channel_indices]  # [B, S, F]
    token_std = target_std[token_channel_indices]    # [B, S, F]

    token_mean = token_mean[:, :, :, None]  # [B, S, F, 1]
    token_std = token_std[:, :, :, None]    # [B, S, F, 1]

    return token_mean, token_std


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

        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        loss = criterion(
            pred=pred,
            target=normalized_targets,
            token_valid_mask=token_valid_mask,
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
        token_inputs = batch["token_inputs"].to(device)
        targets = batch["targets"].to(device)
        token_channel_indices = batch["token_channel_indices"].to(device)
        token_time_indices = batch["token_time_indices"].to(device)
        token_valid_mask = batch["token_valid_mask"].to(device)

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

        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        loss = criterion(
            pred=pred,
            target=normalized_targets,
            token_valid_mask=token_valid_mask,
        )

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
