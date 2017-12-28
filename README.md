# text-classification-extra

支持向量机文本分类器，利用liblinear实现[HanLP](https://github.com/hankcs/HanLP)中的文本分类接口。由于HanLP是个独立项目，并不依赖第三方类库，所以将这个额外的分类器放在这里。

用法参考`com.hankcs.hanlp.classification.classifiers.LinearSVMClassifierTest`，与HanLP的接口完全兼容。

