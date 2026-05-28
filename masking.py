import numpy as np
import torch


def generate_block_mask(
    num_tokens: int,
    mask_ratio: float,
    min_block_tokens: int,
    max_block_tokens: int,
):
    """
    生成 token mask。

    Returns:
        mask: [num_tokens]

    其中:
        1 表示该 token 被 mask
        0 表示该 token 保留
    """
    if num_tokens <= 0:
        return np.zeros((0,), dtype=np.float32)

    target_masked = int(round(num_tokens * mask_ratio))
    target_masked = max(0, min(num_tokens, target_masked))

    mask = np.zeros(num_tokens, dtype=np.float32)
    masked = 0

    while masked < target_masked:
        block = np.random.randint(min_block_tokens, max_block_tokens + 1)

        start = np.random.randint(
            0,
            max(1, num_tokens - block + 1),
        )

        end = min(num_tokens, start + block)

        newly_masked = (mask[start:end] == 0).sum()

        mask[start:end] = 1.0

        masked += newly_masked

        if mask.sum() >= target_masked:
            break

    return mask


def apply_token_mask(tokens: torch.Tensor, token_mask: torch.Tensor):
    """
    在 token 维度上 mask。

    Args:
        tokens: [B, S, L]
        token_mask: [B, S]

    Returns:
        masked_tokens: [B, S, L]
    """
    masked_tokens = tokens.clone()

    masked_tokens[token_mask.bool()] = 0.0

    return masked_tokens