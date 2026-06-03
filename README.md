# EEG ViT-style Self-Supervised Pretraining

首先要构造原始数据集，不论原始数据多长，多按照clip_seconds和clip_stride_seconds对数据进行切割，比如一个被试100s的EEG，设定一个样本长度10s，切割后会得到10个样本。

然后对于每个样本计算时频图，同时对原始数据和时频图按通道进行标准化处理。这时候会得到（64，T）和（64，5，T）两组数据。

然后再按照设定的patch_seconds对样本进行分patch，从而可以输入到ViT。

在每个step，会随机mask一些token（这里已经改回随机mask，而不是固定mask），然后训练模型把对应的完整样本的时频图重构出来。因为之前已经对时频图按通道标准化，所以这里不存在低频主导的问题。