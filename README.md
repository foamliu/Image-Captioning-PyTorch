# 图像中文描述

图像中文描述 + 视觉注意力的 PyTorch 实现。

[Show, Attend, and Tell](https://arxiv.org/pdf/1502.03044.pdf) 是令人惊叹的工作，[这里](https://github.com/kelvinxu/arctic-captions)是作者的原始实现。

这个模型学会了“往哪瞅”：当模型逐词生成标题时，模型的目光在图像上移动以专注于跟下一个词最相关的部分。

## 依赖
- Python 3.5
- PyTorch 0.4

## 数据集

使用 AI Challenger 2017 的图像中文描述数据集，包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。


 ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/dataset.png)

下载点这里：[图像中文描述数据集](https://challenger.ai/datasets/)，放在 data 目录下。


## 网络结构

 ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/net.png)

## 用法

### 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre_process.py
```

### 训练
```bash
$ python train.py
```

可视化训练过程，执行：
```bash
$ tensorboard --logdir path_to_current_dir/logs
```

### 演示
下载 [预训练模型](https://github.com/foamliu/Image-Captioning-v2/releases/download/v1.0/BEST_checkpoint_.pth.tar) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

原图 | 注意力 |
|---|---|
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_0.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_0.jpg) | 
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_1.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_1.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_2.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_2.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_3.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_3.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_4.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_4.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_5.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_5.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_6.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_6.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_7.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_7.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_8.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_8.jpg) |
|<img src="https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_9.jpg" width="400" />|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_9.jpg) |

## 小小的赞助~
<p align="center">
	<img src="https://github.com/foamliu/Image-Captioning-PyTorch/blob/master/sponsor.jpg" alt="Sample"  width="324" height="504">
	<p align="center">
		<em>若对您有帮助可给予小小的赞助~</em>
	</p>
</p>
<br/><br/><br/>

