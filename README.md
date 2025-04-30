# 基于深度学习的图像风格迁移算法研究

*项目基本开发完成，可以进行模型训练、推演和评估。如有问题联系邮箱 fingsinz@foxmail.com*

*此仓库为 PyTorch实现。*

## 一、前言

*基于深度学习的图像风格迁移算法研究——毕业设计*

本项目研究了基于 MetaNet 的图像风格迁移算法，并对该算法进行一定的改进。

网页文档: [https://fingsinz.github.io/StyleTransfer/](https://fingsinz.github.io/StyleTransfer/)

本项目结构：

```
StyleTransfer-PyTorch/
├── README.md                   # README
├── datasets/                   # 训练数据集
├── output/                     # 推演程序、评估程序输出
├── results/                    # 训练结果 png & pth
├── script/                     # 获取数据集的脚本
├── src/                        # 源码
│   ├── data/                   # 数据集加载模块
│   ├── models/                 # 模型定义模块
│   ├── utils/                  # 工具模块
│   ├── config.yaml             # 训练模块配置文件      (训练必须配置)
│   ├── config_example.yaml     # 训练模块配置文件示例
│   ├── inference.py            # 迁移程序
│   ├── train.py                # 训练程序
│   ├── batch_eval.py           # 批量评估性能程序
└── requirements.txt            # 环境依赖
```

## 二、快速上手

### 2.1 安装环境

1. 克隆仓库：

```bash
git clone https://github.com/Fingsinz/StyleTransfer-PyTorch.git
```

2. 配置 Python 环境，安装依赖包:
    - 如果是已有 PyTorch-Cuda 环境，只需安装 `tqdm`、`swanlab`、`scikit-learn`、`scikit-image` 即可。
    - 否则按照 `requirements.txt` 安装。

### 2.2 准备数据集（`./datasets/`）

*如果不训练模型，可以跳过此步骤。*

可参考 `script/download_style_dataset.sh` 和 `script/download_content_dataset.sh` 脚本自行配置数据集。

- 本代码使用 OpenXLab 中的数据集，其中数据集需要保证内容图像文件夹和风格图像文件夹下直接包含图像文件，无需子文件夹。
    - OpenXLab 中 WikiArt 数据集对风格流派做了分类，需要提取子文件夹中的图像到同一文件夹下，并删除空子文件夹。

### 2.3 训练模型

1. 进入 `./src/` 目录：

```bash
cd ./src
```

2. 拷贝 `config.example.yaml` 文件为 `config.yaml` 进行配置训练超参数：

```bash
cp config.example.yaml config.yaml
```

3. 执行 `python train.py` 进行训练：

```bash
python train.py
```

### 2.4 进行图像风格迁移

*即 Inference 推演*

1. 进入 `./src/` 目录：

```bash
cd ./src
```

2. 在 `config.yaml` 中配置 `MetaNet` 和 `TransformNet` 模型路径。使用以下命令进行推演：

```bash
python inference.py path/to/content_image path/to/style_image
```

- `path/to/content_image` 和 `path/to/style_image` 可以为图像文件，也可以为文件夹。

### 2.5 评估模型性能

1. 进入 `./src/` 目录：

```bash
cd ./src
```

2. 使用以下命令进行评估：

```bash
python batch_eval.py <content_path> <style_path> <transform_path> <mode> [output_name]
```

- 该命令用于小批量评估性能。
- `content_path`：内容图像文件夹。
- `style_path`：风格图像文件夹。
- `transform_path`：迁移后的图像文件夹。
- `mode`：评估模式。1 -> psnr and ssim, 2 -> Gram 余弦相似度, 3 -> FID。
- `output_name`：输出文件名（可选）。

