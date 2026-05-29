"""
标准 EEG 64 通道配置。

这个模板采用常见 10-10 系统通道名。
项目内部统一把 EEG 映射到这个固定顺序。

如果原始 .set 文件缺某些通道：
    - 对应通道补 0
    - channel_valid_mask 对应位置为 0

如果原始通道名是 T3/T4/T5/T6：
    - 自动映射到 T7/T8/P7/P8
"""

STANDARD_64_CHANNELS = [
    "Fp1", "Fpz", "Fp2",
    "AF7", "AF3", "AFz", "AF4", "AF8",
    "F7", "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "Cz", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO7", "PO3", "POz", "PO4", "PO8",
    "O1", "Oz", "O2",
    "Iz",
    "F9", "F10",
]


# 常见别名映射。
CHANNEL_ALIASES = {
    # Old 10-20 temporal names
    "T3": "T7",
    "T4": "T8",
    "T5": "P7",
    "T6": "P8",

    # Sometimes lowercase / uppercase normalized later, but keep here for clarity
    "FP1": "Fp1",
    "FPZ": "Fpz",
    "FP2": "Fp2",
    "FZ": "Fz",
    "CZ": "Cz",
    "PZ": "Pz",
    "OZ": "Oz",
    "AFZ": "AFz",
    "FCZ": "FCz",
    "CPZ": "CPz",
    "POZ": "POz",
}
