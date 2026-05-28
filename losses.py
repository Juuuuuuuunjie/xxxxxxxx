import torch
import torch.nn as nn


class BalancedBandTrajectoryLoss(nn.Module):
    """
    频段均衡的完整片段重构 loss。

    pred / target:
        [B, N, C, F, K]

    其中:
        B = batch size
        N = token 数量
        C = 通道数
        F = 频段数
        K = 每个 token 内部的目标时间帧数

    设计原则:
        1. 对完整片段计算 loss，不只计算被 mask 的 token。
        2. 原始缺失通道不参与 loss。
        3. 每个频段先单独算 loss，再等权平均，避免低频主导。
    """
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, pred, target, channel_valid_mask):
        """
        Args:
            pred: [B, N, C, F, K]
            target: [B, N, C, F, K]
            channel_valid_mask: [B, C]
                1 表示真实存在的通道
                0 表示数据集中缺失、补 0 的通道

        Returns:
            scalar loss
        """
        loss = self.mse(pred, target)  # [B, N, C, F, K]

        valid = channel_valid_mask[:, None, :, None, None]  # [B, 1, C, 1, 1]
        loss = loss * valid

        num_tokens = pred.shape[1]
        num_bands = pred.shape[3]
        frames_per_token = pred.shape[4]

        band_losses = []

        for band_idx in range(num_bands):
            band_loss = loss[:, :, :, band_idx, :]  # [B, N, C, K]

            denom = valid.sum() * num_tokens * frames_per_token
            denom = torch.clamp(denom, min=1.0)

            band_loss = band_loss.sum() / denom
            band_losses.append(band_loss)

        total_loss = torch.stack(band_losses).mean()

        return total_loss
