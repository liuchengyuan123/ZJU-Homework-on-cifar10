# ResNet Training on Cifar10 dataset 

## V1.0: 原始 ResNet50

训练300 epoch后在验证集上最佳准确率0.6685，此后逐渐表现出过拟合。
详细信息查看`checkpoint/Sat-Sep-18-22:52:27-2021`到`ccheckpoint/Tue-Sep-21-19:21:29-2021`下的`remark.md`。

测试集上结果：
|acc|CrossEntropy|
|-|-|
|0.6613|1.799|

## V2.0: 添加dropout层

添加dropout层，设置$p=0.7$。

```python
out = self.dropouts[0](self.layer1(out))
out = self.dropouts[1](self.layer2(out))
out = self.dropouts[2](self.layer3(out))
```

训练200 epoch后在验证集上最佳准确率为0.7493，此后不再明显上升。
详细信息查看`checkpoint/Tue-Sep-21-22:52:22-2021`到`checkpoint/Wed-Sep-22-07:57:06-2021`的`remark.md`。

测试集上结果：

|acc|CrossEntropy|
|-|-|
|0.7436|1.7171|

## 总结

添加dropout层有助于模型减少过拟合，提高泛化能力和预测准确率。