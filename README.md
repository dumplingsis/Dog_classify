# 图像分类项目

本项目基于PyTorch实现了一个图像分类模型，使用ResNet-18/34/50三种模型对狗类图像进行分类。

## 数据集结构

- `dataset/images/Images/`：存放按类别分文件夹的图片，文件夹名为"类别编号-类别名"。
- `dataset/annotations/Annotation/`：存放与图片同名的xml标注文件，包含类别名、图片尺寸、目标框等信息。

## 依赖

- Python 3.6+
- PyTorch 1.10.0+
- torchvision 0.11.0+
- tqdm
- numpy
- Pillow
- scikit-learn
- matplotlib

安装依赖：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 确保数据集已按上述结构放置。
2. 运行主程序：
   ```bash
   python train_classification.py
   ```
3. 训练过程中会自动保存每个模型的最优权重（如`best_resnet18.pth`）和训练曲线图（如`resnet18_curve.png`）。

## 训练流程

- 自动读取数据集，解析xml获取类别名。
- 划分训练、验证、测试集。
- 循环训练ResNet-18、ResNet-34、ResNet-50三种模型。
- 每个模型训练10个epoch，保存验证集准确率最高的权重。
- 训练结束后在测试集上评估模型性能。

## 可视化

训练过程中会自动生成每个模型的loss和accuracy曲线图，保存为`{model_name}_curve.png`。

## 项目结构

- `dataset.py`：数据集读取、解析xml、自定义数据集类、数据增强与预处理。
- `models.py`：模型定义、训练、评估、可视化等。
- `train_classification.py`：主程序，实现训练、验证、测试、可视化流程。 