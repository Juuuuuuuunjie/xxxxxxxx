# EEG ViT-style Self-Supervised Pretraining

这是一个面向多通道 EEG 的 ViT 风格自监督预训练项目。它的基本思路是，先把连续 EEG 统一整理成固定的 64 通道标准模板，再沿时间轴切成很多 channel-time patch token，然后随机 mask 一部分 token，使用 Transformer 根据上下文去重构每个 token 对应的频段能量时间轨迹。

这个项目当前不是做分类任务，而是先做无监督预训练。模型学习的目标不是原始波形本身，而是每个 token 内部多个频段随时间变化的轨迹，因此它会更关注 EEG 的时间结构、通道间关系和频谱动态结构。

目前代码支持直接读取 EEGLAB 的 `.set` 数据。读取后会先进行通道名清洗和标准化，并映射到固定顺序的 64 个标准 EEG 通道。缺失通道会自动补 0，同时保留有效通道 mask，避免无效通道参与监督。之后会对连续 EEG 进行切片，生成多个固定长度样本，再进一步切成 ViT 风格的 patch token，并为每个 token 构造对应的时频目标。

模型输入的基本形式是 `[B, S, L]`，其中 `B` 是 batch size，`S` 是 token 数量，`L` 是每个 token 的时间长度。模型输出是 `[B, S, F, K]`，表示每个 token 在多个频段、多个时间点上的能量轨迹预测。训练时会对 token 序列做 mask，模型虽然看不到被遮挡 token 的输入，但仍然需要对所有有效 token 输出重构结果。

当前版本的主入口是 `train_real_set.py`。运行后会完成 `.set` 文件读取、样本切分、标准 64 通道映射、token 与 target 构造、模型训练、checkpoint 保存，以及训练集上的重构可视化。可视化结果会展示 masked token 和 unmasked token 的重构效果，方便直接检查模型是否学到了合理的频段轨迹结构。

项目中的主要文件包括：`configs.py` 用于统一管理参数，`channel_config.py` 定义标准 64 通道模板，`preprocessing.py` 负责 EEG 预处理，`io_utils.py` 负责读取和切分 `.set` 数据，`dataset.py` 负责生成模型输入，`targets.py` 负责构造时频轨迹目标，`masking.py` 负责 token mask，`model.py` 定义 Transformer 模型，`losses.py` 定义重构损失，`trainer.py` 负责训练流程，`train_real_set.py` 作为整个项目的运行入口。

如果你想运行这个项目，通常只需要先在 `configs.py` 中设置好 `.set` 文件路径和训练参数，然后执行 `python train_real_set.py` 即可。训练完成后，模型权重会保存在 `checkpoints/`，重构结果图会保存在 `outputs/reconstruction_train/`。
