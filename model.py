import torch
import torch.nn as nn


class PatchTokenizer(nn.Module):
    """
    把每个 channel-time patch 映射成 Transformer token embedding。

    输入:
        [B, S, L]

    输出:
        [B, S, d_model]
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
    Transformer Encoder。
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


class TimeFreqTokenHead(nn.Module):
    """
    对每个 channel-time token 输出频段能量时间轨迹。

    输入:
        [B, S, d_model]

    输出:
        [B, S, F, K]
    """
    def __init__(
        self,
        d_model: int,
        n_bands: int,
        frames_per_patch: int,
    ):
        super().__init__()

        self.n_bands = n_bands
        self.frames_per_patch = frames_per_patch

        output_dim = n_bands * frames_per_patch

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(self, x):
        out = self.head(x)

        B, S, _ = out.shape

        out = out.view(
            B,
            S,
            self.n_bands,
            self.frames_per_patch,
        )

        return out


class EEGPretrainModel(nn.Module):
    """
    ViT-style EEG 预训练模型。

    一个 token = 一个通道上的一个时间 patch。

    输入:
        token_inputs: [B, S, L]
        token_channel_indices: [B, S]
        token_time_indices: [B, S]

    输出:
        pred: [B, S, F, K]
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
        n_time_patches,
        n_bands,
        frames_per_patch,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_time_patches = n_time_patches

        self.tokenizer = PatchTokenizer(
            input_dim=input_dim,
            d_model=d_model,
            dropout=dropout,
        )

        self.channel_embed = nn.Embedding(
            num_embeddings=n_channels,
            embedding_dim=d_model,
        )

        self.time_embed = nn.Embedding(
            num_embeddings=n_time_patches,
            embedding_dim=d_model,
        )

        self.encoder = EEGTransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            depth=depth,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.head = TimeFreqTokenHead(
            d_model=d_model,
            n_bands=n_bands,
            frames_per_patch=frames_per_patch,
        )

    def forward(
        self,
        token_inputs,
        token_channel_indices,
        token_time_indices,
    ):
        """
        Args:
            token_inputs: [B, S, L]
            token_channel_indices: [B, S]
            token_time_indices: [B, S]

        Returns:
            pred: [B, S, F, K]
        """
        x = self.tokenizer(token_inputs)

        ch_pos = self.channel_embed(token_channel_indices)
        time_pos = self.time_embed(token_time_indices)

        x = x + ch_pos + time_pos

        x = self.encoder(x)

        pred = self.head(x)

        return pred
