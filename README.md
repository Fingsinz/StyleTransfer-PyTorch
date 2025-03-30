# 基于深度学习的图像风格迁移算法研究

*此仓库为 PyTorch实现。*

## 前言

*基于深度学习的图像风格迁移算法研究——毕业设计*

网页文档: [https://fingsinz.github.io/StyleTransfer/](https://fingsinz.github.io/StyleTransfer/)

## 快速上手

### 克隆仓库

```bash
git clone https://github.com/Fingsinz/StyleTransfer-PyTorch.git
```

### 准备数据集（`./datasets/`）

可参考 `script/download_style_dataset.sh` 和 `script/download_content_dataset.sh` 脚本自行配置数据集。

### 训练模型（如果需要）

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

### 进行图像风格迁移（如果需要）

*即 Inference 推演*

进入 `./src/` 目录：

```bash
cd ./src
```

确保 `MetaNet` 和 `TransformNet` 模型存在当前目录；

使用以下命令进行推演：

```bash
python inference.py path/to/content_image path/to/style_image
```

### 进行迁移结果打分（如果需要）

*即 Evaluation 评测*

#### 默认风格打分

进入 `./src/` 目录：

```bash
cd ./src
```

修改 `config_example.yaml` 进行参数配置，然后运行以下命令进行打分:

```bash
python evaluation.py path/to/content_image path/to/style_image path/to/result_image
```

#### 其他风格打分

**准备对应风格的数据集，训练风格打分模型。**

进入 `./src/` 目录：

```bash
cd ./src
```

修改 `config_example.yaml` 进行参数配置，然后运行以下命令进行打分:

```bash
python evaluation.py path/to/content_image path/to/style_image path/to/result_image
```
