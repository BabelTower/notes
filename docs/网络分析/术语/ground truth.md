## ground truth
有监督学习数据为(feature,label)的结构，ground truth指的是数据中label是标记正确的，错误的不是。

通常可以把人工标注的称为ground truth，而半自动标注（机器标注）的不是。

如果数据中的label不是ground truth，那么loss的计算将会产生误差，从而影响到模型质量。