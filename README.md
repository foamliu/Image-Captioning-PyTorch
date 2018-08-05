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

1 | 2 | 3 | 4 |
|---|---|---|---|
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/0_image.png)  | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/1_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/2_image.png)| ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/3_image.png) |
|一个 穿着 球衣 的 男人 和 一个 穿着 球衣 的 运动员 在 翠绿 的 球场上 抢 足球 | 室外 有 一个 穿着 深色 上衣 的 男人 在 给 一个 穿着 深色 上衣 的 男人 做 食物 | 一个 穿着 球衣 的 男人 在 运动场 上 打篮球 | 一个 穿着 黑色 裤子 的 男人 坐在 道路 上 的 汽车 上 |
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/4_image.png)  | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/5_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/6_image.png)| ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/7_image.png) |
|球场上 有 三个 穿着 球衣 的 男人 在 踢足球 | 两个 穿着 深色 上衣 的 男人 在 室内 的 桌子 旁 交谈 | 大厅 里 一群 人 的 旁边 有 一个 穿着 黑色 裤子 的 男人 在 看 手机 | 一个 穿着 短裤 的 男人 站 在 海边 的 沙滩 上 |
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/8_image.png)  | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/9_image.png) |![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/10_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/11_image.png)|
|一个 穿着 黑色 裤子 的 男人 和 一个 穿着 裙子 的 女人 行走 在 道路 上 | 一个 右手 拿 着 球拍 的 男人 在 球场上 打网球 | 一群 人 的 旁边 有 一个 穿着 黑色 裤子 的 女人 坐在 室内 的 椅子 上 | 两个 穿着 球衣 的 男人 在 球场上 打篮球 |
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/12_image.png)  | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/13_image.png) |![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/14_image.png)| ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/15_image.png)|
|两个 穿着 球衣 的 女人 在 翠绿 的 草地 上 抢 足球|一个 穿着 黑色 裤子 的 男人 和 一个 穿着 裙子 的 女人 站 在 展板 前|房间 里 有 一个 穿着 黑色 上衣 的 男人 在 给 一个 坐在 椅子 上 的 女人 做 运动|一个 戴着 墨镜 的 女人 和 一个 穿着 黑色 裤子 的 女人 走 在 道路 上|
|![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/16_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/17_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/18_image.png) | ![image](https://github.com/foamliu/Image-Captioning-v2/raw/master/images/19_image.png) |
|一个 穿着 裙子 的 女人 站 在 地 的 道路 上|一个 穿着 白色 上衣 的 女人 坐在 室外 的 空地 上|一个 穿着 球衣 的 男人 在 球场上 踢足球|一个 穿着 深色 上衣 的 男人 和 一个 穿着 浅色 上衣 的 女人 在 草地 上 玩耍|
