import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# 数据集路径
IMG_ROOT = './dataset/images/Images'
ANN_ROOT = './dataset/annotations/Annotation'

# 读取所有类别
CLASSES = sorted(os.listdir(IMG_ROOT))[:30]  # 只保留前10个类别
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
def get_data_paths():
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
    return img_files, ann_files

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