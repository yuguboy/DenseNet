import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.droprate = droprate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.droprate:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        if self.droprate:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, droprate):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.droprate = droprate

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        if self.droprate:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.avg_pool2d(out, 2)


class DenseBlock(nn.Module):
    def __init__(self, n_layers, in_planes, growth_rate, block, droprate=0.0):
        super(DenseBlock, self).__init__()
        self.layers = self._make_layers(block, in_planes, growth_rate, n_layers, droprate)

    @staticmethod
    def _make_layers(block, in_plane, growth_rate, n_layers, droprate):
        layers = []
        for i in range(n_layers):
            layers.append(block(in_plane + i * growth_rate, growth_rate, droprate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class DenseNet3(nn.Module):
    def __init__(self, depth, num_class, growth_rate=12,
                 reduction=0.5, bottleneck=True, droprate=0.0):
        super(DenseNet3, self).__init__()
        in_planes = growth_rate * 2
        n = (depth - 4) / 3
        if bottleneck:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock

        n = int(n)

        # the first conv layer
        self.conv1 = nn.Conv2d(3, in_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        # block 1
        self.block1 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans1 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), droprate)
        in_planes = int(math.floor(in_planes * reduction))

        # block 2
        self.block2 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes + n * growth_rate)
        self.trans2 = TransitionBlock(in_planes, int(math.floor(in_planes * reduction)), droprate)
        in_planes = int(math.floor(in_planes * reduction))

        # block 3
        self.block3 = DenseBlock(n, in_planes, growth_rate, block, droprate)
        in_planes = int(in_planes + n * growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_planes, num_class)
        self.in_planes = in_planes

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_planes)
        return self.fc(out)