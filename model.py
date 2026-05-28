import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """
    把一个 EEG 时间窗 token 从原始时域输入映射到 Transformer 的 d_model 维度。

    输入:
        [B, N, input_dim]

    输出:
        [B, N, d_model]
    """
    def __init__(self, input_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()

        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.proj(x)


class EEGTransformerEncoder(nn.Module):
    """
    Transformer Encoder，用于建模不同时间 token 之间的上下文关系。
    """
    def __init__(
        self,
        d_model=256,
        n_heads=8,
        depth=6,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

    def forward(self, x):
        return self.encoder(x)


class TimeFreqTrajectoryHead(nn.Module):
    """
    输出每个 token 内部的频段能量时间轨迹。

    输入:
        [B, N, d_model]

    输出:
        [B, N, C, BANDS, K]

    其中:
        C = EEG 通道数
        BANDS = 频段数
        K = 每个 token 内部的目标时间帧数
    """
    def __init__(
        self,
        d_model: int,
        n_channels: int,
        n_bands: int,
        frames_per_token: int,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_bands = n_bands
        self.frames_per_token = frames_per_token

        output_dim = n_channels * n_bands * frames_per_token

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        out = self.head(x)

        B, N, _ = out.shape

        out = out.view(
            B,
            N,
            self.n_channels,
            self.n_bands,
            self.frames_per_token,
        )

        return out


class EEGPretrainModel(nn.Module):
    """
    EEG 预训练模型。

    输入:
        token_inputs: [B, N, D]

    输出:
        pred: [B, N, C, BANDS, K]
    """
    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        depth,
        mlp_ratio,
        dropout,
        n_channels,
        n_bands,
        frames_per_token,
        max_tokens=1024,
    ):
        super().__init__()

        self.tokenizer = PatchTokenizer(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.pos_embed = nn.Parameter(
            torch.randn(1, max_tokens, d_model) * 0.02
        )

        self.encoder = EEGTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.head = TimeFreqTrajectoryHead(
            d_model=d_model,
            n_channels=n_channels,
            n_bands=n_bands,
            frames_per_token=frames_per_token,
        )

    def forward(self, token_inputs):
        """
        Args:
            token_inputs: [B, N, D]

        Returns:
            pred: [B, N, C, BANDS, K]
        """
        B, N, D = token_inputs.shape

        x = self.tokenizer(token_inputs)

        x = x + self.pos_embed[:, :N, :]

        x = self.encoder(x)

        pred = self.head(x)

        return pred
