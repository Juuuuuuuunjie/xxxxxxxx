import torch
import torch.nn as nn


class BalancedBandTokenTrajectoryLoss(nn.Module):
    """
    ViT-style EEG token 重构 loss。

    pred / target:
        [B, S, F, K]

    token_valid_mask:
        [B, S]

    其中:
        B = batch size
        S = token 数量，等于 C * N
        F = 频段数
        K = 每个 token 内部的时频轨迹时间点数量

    设计:
        1. 在所有 token 上计算 loss。
        2. 原始缺失通道对应的 token 不参与 loss。
        3. 每个频段单独算 loss，再等权平均，避免低频主导。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        pred,
        target,
        token_valid_mask,
    ):
        """
        Args:
            pred: [B, S, F, K]
            target: [B, S, F, K]
            token_valid_mask: [B, S]

        Returns:
            scalar loss
        """
        loss = self.mse(pred, target)  # [B, S, F, K]

        valid = token_valid_mask[:, :, None, None]  # [B, S, 1, 1]
        loss = loss * valid

        num_bands = pred.shape[2]
        frames_per_patch = pred.shape[3]

        band_losses = []

        for band_idx in range(num_bands):
            band_loss = loss[:, :, band_idx, :]  # [B, S, K]

            denom = valid.sum() * frames_per_patch
            denom = torch.clamp(denom, min=1.0)

            band_loss = band_loss.sum() / denom
            band_losses.append(band_loss)

        total_loss = torch.stack(band_losses).mean()

        return total_loss
