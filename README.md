# EEG ViT-style Self-Supervised Pretraining

首先要构造原始数据集，不论原始数据多长，多按照clip_seconds和clip_stride_seconds对数据进行切割，比如一个被试100s的EEG，设定一个样本长度10s，切割后会得到10个样本。

然后对于每个样本计算时频图，同时对原始数据和时频图按通道进行标准化处理。这时候会得到（64，T）和（64，5，T）两组数据。

然后再按照设定的patch_seconds对样本进行分patch，从而可以输入到ViT。

在每个step，会随机mask一些token（这里已经改回随机mask，而不是固定mask），然后训练模型把对应的完整样本的时频图重构出来。因为之前已经对时频图按通道标准化，所以这里不存在低频主导的问题。

---

更新了convert_set_to_lance.py和lance_dataset.py，可以把原始数据处理好并转成lance格式，主程序直接读取lance格式数据处理。

---

为了能用上服务器的数据，同时避免一边做时频变换一边训练，现在主要把服务器的数据过一遍我们的预处理，然后再做预训练。

为了实现这一点：

* 新增了build_processed_lance_from_raw.py，主要用于把服务器的lance数据，完整走一遍我们的预处理，生成一组输入输出的数据，输入是我们处理过的EEG，输出是EEG对应的时频图。

* checkTransformedLanceData.py可以用于检查处理后的lance数据是否正常

