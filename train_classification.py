import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from dataset import get_data_paths, DogDataset, transform_train, transform_test, CLASSES
from models import get_model, train_one_epoch, evaluate, plot_history, MODEL_NAMES

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
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    # 训练、验证、测试
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(CLASSES)
    num_epochs = 50
    os.makedirs('result', exist_ok=True)
    for model_name in MODEL_NAMES:
        print(f'\n==== 训练模型: {model_name} ====')
        model = get_model(model_name, num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        best_acc = 0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        # 创建子文件夹，名称为模型名称和训练开始时间
        start_time = time.strftime('%Y%m%d_%H%M%S')
        result_dir = os.path.join('result', f'{model_name}_{start_time}')
        os.makedirs(result_dir, exist_ok=True)
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
                torch.save(model.state_dict(), os.path.join(result_dir, f'best_{model_name}.pth'))
        plot_history(history, model_name, result_dir)
        # 测试集评估
        model.load_state_dict(torch.load(os.path.join(result_dir, f'best_{model_name}.pth')))
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'==== {model_name} 测试集准确率: {test_acc:.4f} ====')

if __name__ == '__main__':
    main()
