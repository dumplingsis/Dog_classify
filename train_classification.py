import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from dataset import get_data_paths, DogDataset, transform_train, transform_test, CLASSES
from models import get_model, train_one_epoch, evaluate, plot_history, MODEL_NAMES

# 数据集路径
IMG_ROOT = './dataset/images/Images'
ANN_ROOT = './dataset/annotations/Annotation'

# 读取所有类别
CLASSES = sorted(os.listdir(IMG_ROOT))
CLASS_TO_IDX = {cls: idx for idx, cls in enumerate(CLASSES)}

# 解析xml获取类别名
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    name = root.find('object').find('name').text
    return name

# 自定义数据集
class DogDataset(Dataset):
    def __init__(self, img_paths, ann_paths, transform=None):
        self.img_paths = img_paths
        self.ann_paths = ann_paths
        self.transform = transform
        self.labels = []
        for ann in ann_paths:
            name = parse_xml(ann)
            for folder in CLASSES:
                if name in folder:
                    self.labels.append(CLASS_TO_IDX[folder])
                    break

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# 获取所有图片和标注路径
img_files = []
ann_files = []
for cls in CLASSES:
    img_dir = os.path.join(IMG_ROOT, cls)
    ann_dir = os.path.join(ANN_ROOT, cls)
    imgs = glob.glob(os.path.join(img_dir, '*.jpg'))
    for img_path in imgs:
        base = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(ann_dir, base)
        if os.path.exists(ann_path):
            img_files.append(img_path)
            ann_files.append(ann_path)

# 划分训练、验证、测试集
train_imgs, test_imgs, train_anns, test_anns = train_test_split(img_files, ann_files, test_size=0.2, random_state=42)
train_imgs, val_imgs, train_anns, val_anns = train_test_split(train_imgs, train_anns, test_size=0.1, random_state=42)

# 数据增强与预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = DogDataset(train_imgs, train_anns, transform=transform_train)
val_dataset = DogDataset(val_imgs, val_anns, transform=transform_test)
test_dataset = DogDataset(test_imgs, test_anns, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

def main():
    # 获取所有图片和标注路径
    img_files, ann_files = get_data_paths()
    # 划分训练、验证、测试集
    train_imgs, test_imgs, train_anns, test_anns = train_test_split(img_files, ann_files, test_size=0.2, random_state=42)
    train_imgs, val_imgs, train_anns, val_anns = train_test_split(train_imgs, train_anns, test_size=0.1, random_state=42)
    # 构建数据集
    train_dataset = DogDataset(train_imgs, train_anns, transform=transform_train)
    val_dataset = DogDataset(val_imgs, val_anns, transform=transform_test)
    test_dataset = DogDataset(test_imgs, test_anns, transform=transform_test)
    # 构建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    # 训练、验证、测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(CLASSES)
    num_epochs = 10
    for model_name in MODEL_NAMES:
        print(f'\n==== 训练模型: {model_name} ====')
        model = get_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}')
            print(f'Val   Loss: {val_loss:.4f}, Acc: {val_acc:.4f}')
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'best_{model_name}.pth')
        plot_history(history, model_name)
        # 测试集评估
        model.load_state_dict(torch.load(f'best_{model_name}.pth'))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'==== {model_name} 测试集准确率: {test_acc:.4f} ====')

if __name__ == '__main__':
    main()
