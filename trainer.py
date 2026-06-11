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

    return torch.as_tensor(
        masks,
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
        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]                    # [B, S, L]
        targets = batch["targets"]                              # [B, S, F, K]
        token_channel_indices = batch["token_channel_indices"]  # [B, S]
        token_time_indices = batch["token_time_indices"]        # [B, S]
        token_valid_mask = batch["token_valid_mask"]            # [B, S]

        B, S, _ = token_inputs.shape

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

        optimizer.zero_grad(set_to_none=True)
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
        batch = move_batch_to_device(batch, device)

        token_inputs = batch["token_inputs"]
        targets = batch["targets"]
        token_channel_indices = batch["token_channel_indices"]
        token_time_indices = batch["token_time_indices"]
        token_valid_mask = batch["token_valid_mask"]

        B, S, _ = token_inputs.shape

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
