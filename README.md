#### 使用的模型的GitHub链接
[https://github.com/PaddlePaddle/PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)

## 数据标注

### 标注工具：PPOCRLabel

#### 安装方法：
首先参考以下文档安装PaddlePaddle：
[https://www.paddlepaddle.org.cn/install/quick](https://www.paddlepaddle.org.cn/install/quick)

然后安装PPOCRLabel：
```bash
pip install PPOCRLabel
```

### 标注方法:
首先点击左下角的自动标注，由模型先对所有图片进行一次自动标注后手动调整。  
手动调整方法：在图中拖动和缩放检测框以手动框选完整文本区域；在右边中间的识别结果处可以手动调整模型的识别结果；在右上角处点击矩形标注或多点标注，可以手动框选模型未检测到的部分。  
如果确认图片内检测框和文本信息无误，点击确认即可开始标注下一张图片。全部标注完成后，点击文件-导出标记结果和导出识别结果，即可在图片文件夹路径下导出对应的标注文件。

## 算法模型
使用PaddleOCR模型，如果在上一步已经安装好了PaddlePaddle，则仅需运行下面的命令即可安装PaddleOCR whl包
```bash
pip install paddleocr
```
大致原理:  
从整体结构上看，PP-OCRv4仍采取先检测后识别的两阶段方法，使用可微二值化（Differentiable Binarization, DB）算法（论文地址[https://arxiv.org/pdf/1911.08947](https://arxiv.org/pdf/1911.08947)）做文本检测，然后使用MobileNetv3做方向分类以应对不同方向的文本识别，然后使用STVR（论文地址[https://arxiv.org/abs/2205.00159](https://arxiv.org/abs/2205.00159)）进行文本识别。

## 测试方法
软件环境参考上述环境配置方法，在AMD Ryzen 7 5800H with Radeon Graphics，NVIDIA GeForce RTX 3060硬件环境下，在自行建立的测试数据集上（有效图片数326，多为短视频截图）测试，将模型识别的结果和手动标注均拼接成完整的字符串，计算两个字符串的词错率（Word Error Rate, WER）、准确率和CPU利用率等性能，结果如下：  
WER：0.142（准确率：85.8%）  
CPU Usage：22.1%  
Memory Usage：10867.4MB  
GPU Usage：1411.1MB  
Peak Memory Usage：56.2MB  
Time：54.2s（每秒处理图片数5.9张）  
注:由于手动添加标注的时候顺序会和实际识别的时候顺序有区别,会导致人为因素的准确率降低,排除这一干扰后实际准确率约97%
