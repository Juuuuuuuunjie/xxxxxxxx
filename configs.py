from dataclasses import dataclass, field


@dataclass
class DataConfig:
    target_sfreq: int = 100
    n_channels: int = 64
    clip_seconds: int = 10

    # Token 切分方式
    window_seconds: float = 1.0
    window_stride_seconds: float = 1.0

    # 时频目标的时间分辨率
    # 例如 0.1 秒表示每个 token 内部重构 10 个时间点的频段能量轨迹
    target_frame_seconds: float = 0.1

    # STFT 窗长
    # 为了估计低频，比如 delta 1-4 Hz，窗长不能太短
    stft_window_seconds: float = 1.0

    # STFT hop
    # 一般和 target_frame_seconds 保持一致
    stft_hop_seconds: float = 0.1

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
    mask_ratio: float = 0.5
    min_block_tokens: int = 1
    max_block_tokens: int = 3


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
    save_path: str = "pretrain_minimal.pt"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
