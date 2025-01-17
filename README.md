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
安装好后通过以下命令运行PPOCRLabel
```bash
PPOCRLabel --lang ch
```
在文件-打开目录中选择需标注数据集的图片文件目录然后点击左下角的自动标注，由模型先对所有图片进行一次自动标注后手动调整。  

手动调整方法：在图中拖动和缩放检测框以手动框选完整文本区域；在右边中间的识别结果处可以手动调整模型的识别结果；在右上角处点击矩形标注或多点标注，可以手动框选模型未检测到的部分。 

如果确认图片内检测框和文本信息无误，点击确认即可开始标注下一张图片。全部标注完成后，点击文件-导出标记结果和导出识别结果，即可在图片文件夹路径下导出对应的标注文件和crops。

## 算法模型
使用PaddleOCR模型，如果在上一步已经安装好了PaddlePaddle，则仅需运行下面的命令即可安装PaddleOCR whl包
```bash
pip install paddleocr
```

大致原理: 

![img_1](https://github.com/user-attachments/assets/8894ee4f-f698-442b-95fc-aaf408af90f2)

从整体结构上看，PP-OCRv4仍采取先检测后识别的两阶段方法，使用可微二值化（Differentiable Binarization, DB）算法（论文地址[https://arxiv.org/pdf/1911.08947](https://arxiv.org/pdf/1911.08947)）进行文本检测，然后使用MobileNetv3进行方向分类以应对不同方向的文本识别，然后使用STVR（论文地址[https://arxiv.org/abs/2205.00159](https://arxiv.org/abs/2205.00159)）进行文本识别。

更详细的算法及模型原理请参考[https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/algorithm_overview.md](https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_ch/algorithm_overview.md)

## 测试方法
软件环境参考上述环境配置方法，在AMD Ryzen 7 5800H with Radeon Graphics，NVIDIA GeForce RTX 3060硬件环境下测试 

测试代码参考test.py，使用时请将其中的图片文件夹路径和标注路径替换为自己的对应路径  
加载模型时请根据实际情况选择其中参数，如果不确定实际情况可以换成ocr = PaddleOCR()以使用默认参数

在自行建立的测试数据集上（有效图片数326，多为短视频截图）测试，利用上文中的标注方法进行标注后，将模型识别的结果和标注结果均拼接成完整的字符串，计算两个字符串的词错率（Word Error Rate, WER）、准确率和CPU利用率等性能，结果如下：  
WER：0.125（准确率：87.5%）  
CPU Usage：12.9%  
Memory Usage：13982.1MB  
GPU Usage：1951.5MB  
Peak Memory Usage：123.7MB  
Time：39.3s（每秒处理图片数7.7张）  
注:由于手动添加标注的时候顺序会和实际识别的时候顺序有区别，会导致人为因素的准确率降低，排除这一干扰后实际准确率约97%
