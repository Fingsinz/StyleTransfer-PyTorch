# 基于深度学习的图像风格迁移算法研究

*项目正在开发完善中🏗️，可能存在错误，有问题联系邮箱 fingsinz@foxmail.com*

*此仓库为 PyTorch实现。*

## 一、前言

*基于深度学习的图像风格迁移算法研究——毕业设计*

网页文档: [https://fingsinz.github.io/StyleTransfer/](https://fingsinz.github.io/StyleTransfer/)

本项目结构：

```
StyleTransfer-PyTorch/
├── README.md                   # README
├── datasets/                   # 训练数据集
├── output/                     # inference 程序输出
├── results/                    # 训练结果 png & pth
├── script/                     # 获取数据集的脚本
├── src/                        # 源码
│   ├── data/                   # 数据集加载模块
│   ├── models/                 # 模型定义模块
│   ├── utils/                  # 工具模块
│   ├── config.yaml             # 训练模块配置文件      (训练必须配置)
│   ├── config_example.yaml     # 训练模块配置文件示例
│   ├── config_evaluation.yaml  # 评测模块配置文件
│   ├── evaluation.py           # 评测模块
│   ├── inference.py            # 迁移程序
│   ├── train.py                # 训练程序
│   ├── train_cnn.py            # 基于 CNN 的风格迁移实现
│   └── train_unet.py           # 基于 U-Net 的风格迁移实现
└── requirements.txt            # 环境依赖
```

## 二、快速上手

### 2.1 克隆仓库

```bash
git clone https://github.com/Fingsinz/StyleTransfer-PyTorch.git
```

### 2.2 准备数据集（`./datasets/`）

可参考 `script/download_style_dataset.sh` 和 `script/download_content_dataset.sh` 脚本自行配置数据集。

### 2.3 训练模型

*即 Training 训练*

进入 `./src/` 目录：

```bash
cd ./src
```

拷贝 `config.example.yaml` 文件为 `config.yaml` 进行配置训练超参数：

```bash
cp config.example.yaml config.yaml
```

执行 `python train.py` 进行训练：

```bash
python train.py
```

### 2.4 进行图像风格迁移

*即 Inference 推演*

进入 `./src/` 目录：

```bash
cd ./src
```

确保 `MetaNet` 和 `TransformNet` 模型存在当前目录，在 `src/inference.py` 中加载对应的模型。使用以下命令进行推演：

```bash
python inference.py path/to/content_image path/to/style_image
```

