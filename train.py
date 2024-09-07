import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import TwoStreamAGCN
from data_loader import load_data

# 加载数据
train_loader = load_data(
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_joint.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_bone.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\train_label.npy',
    batch_size=8
)

test_loader = load_data(
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_joint.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_bone.npy',
    'D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\data\\test_label.npy',
    shuffle=False
)

# 初始化模型
A = torch.eye(17, dtype=torch.float32)  # 邻接矩阵
model_joint = TwoStreamAGCN(num_class=155, num_point=17, num_person=2, in_channels=3, A=A, dropout_rate=0.5).cuda()
model_bone = TwoStreamAGCN(num_class=155, num_point=17, num_person=2, in_channels=3, A=A, dropout_rate=0.5).cuda()

# 定义损失函数和优化器（包含 L2 正则化）
criterion = nn.CrossEntropyLoss()
optimizer_joint = torch.optim.Adam(model_joint.parameters(), lr=0.001, weight_decay=1e-4)
optimizer_bone = torch.optim.Adam(model_bone.parameters(), lr=0.001, weight_decay=1e-4)

# 学习率调度器
scheduler_joint = ReduceLROnPlateau(optimizer_joint, mode='max', patience=5, factor=0.5, verbose=True)
scheduler_bone = ReduceLROnPlateau(optimizer_bone, mode='max', patience=5, factor=0.5, verbose=True)

# 初始化 TensorBoard
writer = SummaryWriter(log_dir='runs/experiment_name')

# 训练模型
def train_model(model_joint, model_bone, train_loader, test_loader, criterion, optimizer_joint, optimizer_bone, scheduler_joint, scheduler_bone, epochs=100):
    best_accuracy = 0.0  # 初始化最佳准确率

    for epoch in range(epochs):
        model_joint.train()
        model_bone.train()

        running_loss = 0.0
        correct = 0
        total = 0
        for i, (joint, bone, labels) in enumerate(train_loader):
            joint, bone, labels = joint.cuda(), bone.cuda(), labels.cuda()

            # 前向传播
            output_joint = model_joint(joint)
            output_bone = model_bone(bone)
            output = (output_joint + output_bone) / 2
            loss = criterion(output, labels)

            # 反向传播
            optimizer_joint.zero_grad()
            optimizer_bone.zero_grad()
            loss.backward()
            optimizer_joint.step()
            optimizer_bone.step()

            running_loss += loss.item()

            # 计算训练集精度
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 100 == 99:
                writer.add_scalar('Training Loss', running_loss / 100, epoch * len(train_loader) + i)
                running_loss = 0.0

        # 记录每个 epoch 的训练集精度
        train_accuracy = 100 * correct / total
        writer.add_scalar('Training Accuracy', train_accuracy, epoch)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Training Accuracy: {train_accuracy:.2f}%')

        # 每个 epoch 后进行测试并记录准确率
        test_accuracy, confidences = test_model_with_confidence(model_joint, model_bone, test_loader)

        writer.add_scalar('Test Accuracy', test_accuracy, epoch)

        # 调整学习率
        scheduler_joint.step(test_accuracy)
        scheduler_bone.step(test_accuracy)

        # 如果当前的准确率比之前的最高准确率高，保存置信度文件
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            np.save('D:\\AI Competition\\基于无人机的人体行为识别-参赛资源(省赛)\\eval\\pred.npy', confidences)
            print(f'保存了新的最佳置信度文件！测试准确率: {best_accuracy:.2f}%')

    writer.close()

# 测试模型并返回置信度和准确率
def test_model_with_confidence(model_joint, model_bone, test_loader):
    model_joint.eval()
    model_bone.eval()
    correct = 0
    total = 0
    all_confidences = []

    with torch.no_grad():
        for batch_data in test_loader:
            joint, bone, labels = batch_data
            joint, bone, labels = joint.cuda(), bone.cuda(), labels.cuda()

            # 前向传播
            output_joint = model_joint(joint)
            output_bone = model_bone(bone)

            # 融合输出
            output = (output_joint + output_bone) / 2

            # 计算置信度
            confidence = torch.softmax(output, dim=1)
            all_confidences.append(confidence.cpu().numpy())

            # 计算预测类别
            _, predicted = torch.max(output, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    all_confidences = np.vstack(all_confidences)

    return accuracy, all_confidences


# 训练和测试
train_model(model_joint, model_bone, train_loader, test_loader, criterion, optimizer_joint, optimizer_bone, scheduler_joint, scheduler_bone, epochs=100)

