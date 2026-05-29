from dataclasses import dataclass, field


@dataclass
class DataConfig:
    """
    数据相关配置。
    """

    # 你的 .set 文件路径。
    # 可以是单个 .set 文件，也可以是多个 .set 文件。
    set_paths: list = field(default_factory=lambda: [
        "data/sub01_01_EC.set",
    ])

    # 是否把 MNE 读出来的 Volt 转成 microvolt。
    # MNE 读 EEG 通常单位是 Volt，数值很小，例如 1e-5。
    # 转成 uV 后更适合模型训练。
    convert_v_to_uv: bool = True

    # 每个训练样本切多长，单位秒。
    clip_seconds: float = 10.0

    # 连续 EEG 切样本时的步长，单位秒。
    # 如果等于 clip_seconds，就是无重叠切分。
    # 如果小于 clip_seconds，就是有重叠切分。
    clip_stride_seconds: float = 10.0

    # 目标采样率。
    target_sfreq: float = 100.0

    # 固定标准通道数量。
    n_channels: int = 64

    # 每个 token 对应多长 EEG，单位秒。
    patch_seconds: float = 1.28

    # patch 滑动步长，单位秒。
    # 如果和 patch_seconds 相等，就是不重叠 patch。
    patch_stride_seconds: float = 1.28

    # 每个 patch 内部的 target 时间分辨率。
    # 例如 patch_seconds=1.28, target_frame_seconds=0.16
    # 则每个 patch 内有 8 个 target frame。
    target_frame_seconds: float = 0.16

    # STFT 参数。
    stft_window_seconds: float = 1.0
    stft_hop_seconds: float = 0.16

    # 频段定义。
    # 你也可以按自己任务改。
    band_defs: list = field(default_factory=lambda: [
        ("delta", 1.0, 4.0),
        ("theta", 4.0, 8.0),
        ("alpha", 8.0, 13.0),
        ("beta", 13.0, 30.0),
        ("gamma_low", 30.0, 45.0),
    ])

    eps: float = 1e-6


@dataclass
class MaskConfig:
    """
    mask 相关配置。
    """
    mask_ratio: float = 0.4
    min_block_tokens: int = 4
    max_block_tokens: int = 32


@dataclass
class ModelConfig:
    """
    模型相关配置。
    """
    d_model: int = 256
    n_heads: int = 8
    depth: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1


@dataclass
class TrainConfig:
    """
    训练相关配置。
    """
    seed: int = 42
    device: str = "cuda"

    batch_size: int = 8
    num_workers: int = 0

    # 你现在希望训练 10 个 epoch。
    num_epochs: int = 10

    lr: float = 1e-4
    weight_decay: float = 1e-4

    print_every: int = 10

    save_path: str = "checkpoints/pretrain_real_set_best.pt"

    # 训练结束后保存可视化图像的位置。
    vis_dir: str = "outputs/reconstruction_train"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    mask: MaskConfig = field(default_factory=MaskConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
