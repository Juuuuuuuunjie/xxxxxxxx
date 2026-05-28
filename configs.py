from dataclasses import dataclass, field


@dataclass
class DataConfig:
    # 统一采样率
    target_sfreq: int = 100

    # 固定 EEG 通道数
    n_channels: int = 64

    # 每个样本长度，单位秒
    # 如果 target_sfreq=100, clip_seconds=10, 那么 T=1000
    clip_seconds: int = 10

    # 每个 token 的时间长度，单位秒
    # 例如 target_sfreq=100, patch_seconds=1.28, 则 patch_len=128
    patch_seconds: float = 1.28

    # token 的时间步长，单位秒
    # 如果等于 patch_seconds，则是不重叠切分
    patch_stride_seconds: float = 1.28

    # 每个 token 内部要重构多少个时频轨迹点
    # 例如 patch_seconds=1.28, target_frame_seconds=0.16, 则 K=8
    target_frame_seconds: float = 0.16

    # STFT 窗长
    # 为了估计低频，不能太短
    stft_window_seconds: float = 1.0

    # STFT hop
    stft_hop_seconds: float = 0.08

    # 频段定义
    band_defs: tuple = (
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma_low", 30.0, 40.0),
    )

    eps: float = 1e-6


@dataclass
class MaskConfig:
    # 在 [B, C*N, L] 的 token 序列上随机 mask
    mask_ratio: float = 0.5

    # block mask 的最短 token 数
    min_block_tokens: int = 1

    # block mask 的最长 token 数
    max_block_tokens: int = 8


@dataclass
class ModelConfig:
    d_model: int = 256
    n_heads: int = 8
    depth: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    batch_size: int = 8
    num_epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-4
    device: str = "cuda"
    seed: int = 42
    num_workers: int = 0
    print_every: int = 10
    save_path: str = "pretrain_vit_eeg.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
