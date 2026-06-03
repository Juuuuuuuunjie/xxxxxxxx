import numpy as np
import torch

from masking import generate_block_mask, apply_token_mask


def build_batch_token_mask(batch_size, num_tokens, cfg, device):
    masks = []

    for _ in range(batch_size):
        mask = generate_block_mask(
            num_tokens=num_tokens,
            mask_ratio=cfg.mask.mask_ratio,
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

        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        loss = criterion(
            pred=pred,
            target=targets,
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

        pred = model(
            token_inputs=masked_inputs,
            token_channel_indices=token_channel_indices,
            token_time_indices=token_time_indices,
        )

        loss = criterion(
            pred=pred,
            target=targets,
            token_valid_mask=token_valid_mask,
        )

        total_loss += loss.item()

    return total_loss / max(1, len(loader))
