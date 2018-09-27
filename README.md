# 图像中文描述

图像中文描述 + 视觉注意力

## 依赖
- [NumPy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
- [Tensorflow](https://www.tensorflow.org/versions/r0.8/get_started/os_setup.html)
- [Keras](https://keras.io/#installation)
- [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/)

## 数据集

使用 AI Challenger 2017 的图像中文描述数据集，包含30万张图片，150万句中文描述。训练集：210,000 张，验证集：30,000 张，测试集 A：30,000 张，测试集 B：30,000 张。


 ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/dataset.png)

下载点这里：[图像中文描述数据集](https://challenger.ai/datasets/caption)，放在 data 目录下。


## 网络结构

 ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/net.png)

## 用法

### 数据预处理
提取210,000 张训练图片和30,000 张验证图片：
```bash
$ python pre-process.py
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
下载 [预训练模型](https://github.com/foamliu/Image-Captioning-v2/releases/download/v1.0/model.85-0.7657.hdf5) 放在 models 目录，然后执行:

```bash
$ python demo.py
```

原图 | 注意力 |
|---|---|
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/image_0.jpg){width=200px}|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/out_0.jpg) | 
