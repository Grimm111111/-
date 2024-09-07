import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(GraphConvolution, self).__init__()
        self.A = nn.Parameter(A, requires_grad=True)  # 直接将 A 作为 nn.Parameter
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device)  # 将邻接矩阵移动到输入张量所在的设备上
        x = torch.einsum('nctv,vw->nctw', (x, A))
        x = self.conv(x)
        return self.relu(x)

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)

    def forward(self, x):
        avg_out = torch.mean(x, dim=-1, keepdim=True)
        max_out = torch.max(x, dim=-1, keepdim=True)[0]
        out = self.fc1(avg_out + max_out)
        out = self.fc2(F.relu(out))
        return torch.sigmoid(out)


class AGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A):
        super(AGCNBlock, self).__init__()
        self.gcn = GraphConvolution(in_channels, out_channels, A)
        self.att = ChannelAttention(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.gcn(x)
        x = x * self.att(x)  # 使用注意力机制调整通道权重
        return self.relu(x)


class TwoStreamAGCN(nn.Module):
    def __init__(self, num_class, num_point, num_person, in_channels, A, dropout_rate=0.5):
        super(TwoStreamAGCN, self).__init__()
        self.A = A
        self.data_bn = nn.BatchNorm2d(in_channels)

        self.layer1 = AGCNBlock(in_channels, 64, A)
        self.layer2 = AGCNBlock(64, 128, A)
        self.layer3 = AGCNBlock(128, 256, A)

        # 全连接层之前加入 Dropout 防止过拟合
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_class)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x.view(N * M, C, T, V)
        x = self.data_bn(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 全局平均池化
        x = F.avg_pool2d(x, (x.size(2), x.size(3)))  # 输出 (N * M, 256, 1, 1)

        x = x.view(N, -1)  # 展平为 (N, 2048)

        # 在全连接层前加入 Dropout
        x = self.dropout(x)

        return self.fc(x)

