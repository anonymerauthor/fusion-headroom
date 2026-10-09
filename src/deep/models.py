"""Compact CIFAR architectures (32x32 inputs) for the deep-committee experiments."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------- ResNet-18 (He et al. 2016, CIFAR stem)
class BasicBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(cout)
        self.sc = nn.Sequential()
        if stride != 1 or cin != cout:
            self.sc = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        return F.relu(self.b2(self.c2(out)) + self.sc(x))


class ResNet18(nn.Module):
    def __init__(self, nc, width=64):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        layers, cin = [], w
        for cout, s in ((w, 1), (2 * w, 2), (4 * w, 2), (8 * w, 2)):
            layers += [BasicBlock(cin, cout, s), BasicBlock(cout, cout, 1)]
            cin = cout
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(cin, nc)

    def forward(self, x):
        x = self.layers(self.stem(x))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- Wide ResNet-16-4 (Zagoruyko & Komodakis 2016)
class WideBlock(nn.Module):
    def __init__(self, cin, cout, stride):
        super().__init__()
        self.b1 = nn.BatchNorm2d(cin); self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout); self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.sc = None if (cin == cout and stride == 1) else nn.Conv2d(cin, cout, 1, stride, bias=False)

    def forward(self, x):
        o = F.relu(self.b1(x))
        y = self.c2(F.relu(self.b2(self.c1(o))))
        return y + (x if self.sc is None else self.sc(o))


class WRN(nn.Module):
    def __init__(self, nc, depth=16, k=4):
        super().__init__()
        n = (depth - 4) // 6
        ws = [16, 16 * k, 32 * k, 64 * k]
        self.conv = nn.Conv2d(3, ws[0], 3, 1, 1, bias=False)
        blocks, cin = [], ws[0]
        for i, (cout, s) in enumerate(zip(ws[1:], (1, 2, 2))):
            for j in range(n):
                blocks.append(WideBlock(cin, cout, s if j == 0 else 1)); cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.bn = nn.BatchNorm2d(cin); self.fc = nn.Linear(cin, nc)

    def forward(self, x):
        x = F.relu(self.bn(self.blocks(self.conv(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))


# ---------------------------------------------------------------- VGG-11 with batch norm (Simonyan & Zisserman 2015)
class VGG11(nn.Module):
    def __init__(self, nc):
        super().__init__()
        cfg, layers, cin = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"], [], 3
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(2))
            else:
                layers += [nn.Conv2d(cin, v, 3, padding=1, bias=False), nn.BatchNorm2d(v), nn.ReLU()]; cin = v
        self.features = nn.Sequential(*layers); self.fc = nn.Linear(512, nc)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))


# ---------------------------------------------------------------- MobileNetV2 (Sandler et al. 2018), CIFAR strides
class InvRes(nn.Module):
    def __init__(self, cin, cout, expand, stride):
        super().__init__()
        h = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.net = nn.Sequential(nn.Conv2d(cin, h, 1, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
                                 nn.Conv2d(h, h, 3, stride, 1, groups=h, bias=False), nn.BatchNorm2d(h), nn.ReLU6(),
                                 nn.Conv2d(h, cout, 1, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x):
        return x + self.net(x) if self.use_res else self.net(x)


class MobileNetV2(nn.Module):
    CFG = [(1, 16, 1, 1), (6, 24, 2, 1), (6, 32, 3, 2), (6, 64, 4, 2), (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1)]

    def __init__(self, nc):
        super().__init__()
        layers, cin = [nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU6()], 32
        for t, c, n, s in self.CFG:
            for i in range(n):
                layers.append(InvRes(cin, c, t, s if i == 0 else 1)); cin = c
        layers += [nn.Conv2d(cin, 1280, 1, bias=False), nn.BatchNorm2d(1280), nn.ReLU6()]
        self.features = nn.Sequential(*layers); self.fc = nn.Linear(1280, nc)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# ---------------------------------------------------------------- DenseNet-BC (small) (Huang et al. 2017)
class DenseLayer(nn.Module):
    def __init__(self, cin, g):
        super().__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(cin), nn.ReLU(), nn.Conv2d(cin, 4 * g, 1, bias=False),
                                 nn.BatchNorm2d(4 * g), nn.ReLU(), nn.Conv2d(4 * g, g, 3, 1, 1, bias=False))

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


class DenseNet(nn.Module):
    def __init__(self, nc, g=12, blocks=(6, 12, 24)):
        super().__init__()
        c = 2 * g
        layers = [nn.Conv2d(3, c, 3, 1, 1, bias=False)]
        for i, nb in enumerate(blocks):
            for _ in range(nb):
                layers.append(DenseLayer(c, g)); c += g
            if i < len(blocks) - 1:
                co = c // 2
                layers += [nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, co, 1, bias=False), nn.AvgPool2d(2)]; c = co
        layers += [nn.BatchNorm2d(c), nn.ReLU()]
        self.features = nn.Sequential(*layers); self.fc = nn.Linear(c, nc)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# ---------------------------------------------------------------- small plain CNN
class SimpleCNN(nn.Module):
    def __init__(self, nc):
        super().__init__()
        def blk(a, b):
            return nn.Sequential(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2))
        self.f = nn.Sequential(blk(3, 64), blk(64, 128), blk(128, 256))
        self.fc = nn.Linear(256, nc)

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.f(x), 1).flatten(1))


ARCHS = {"resnet18": ResNet18, "wrn16_4": WRN, "vgg11": VGG11, "mobilenetv2": MobileNetV2, "densenet": DenseNet,
         "simplecnn": SimpleCNN}
