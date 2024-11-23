import torch.nn as nn


class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.c11 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.b11 = nn.BatchNorm2d(32)
        self.c12 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.b12 = nn.BatchNorm2d(32)
        self.s1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d1 = nn.Dropout(p=0.2)

        self.c21 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.b21 = nn.BatchNorm2d(64)
        self.c22 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.b22 = nn.BatchNorm2d(64)
        self.s2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d2 = nn.Dropout(p=0.3)

        self.c31 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.b31 = nn.BatchNorm2d(128)
        self.c32 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.b32 = nn.BatchNorm2d(128)
        self.s3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.d3 = nn.Dropout(p=0.4)

        # self.f4 = nn.Linear(128*8*8, 128)
        # self.b4 = nn.BatchNorm1d(128)
        # self.d4 = nn.Dropout(p=0.5)

        self.output = nn.Linear(128*8*8, 200)

        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.c11(x))
        x = self.b11(x)
        x = self.act(self.c12(x))
        x = self.b12(x)
        x = self.s1(x)
        x = self.d1(x)

        x = self.act(self.c21(x))
        x = self.b21(x)
        x = self.act(self.c22(x))
        x = self.b22(x)
        x = self.s2(x)
        x = self.d2(x)

        x = self.act(self.c31(x))
        x = self.b31(x)
        x = self.act(self.c32(x))
        x = self.b32(x)
        x = self.s3(x)
        x = self.d3(x)

        x = x.view(-1, x.size(1)*x.size(2)*x.size(3))
        # x = self.act(self.f4(x))
        # x = self.b4(x)
        # x = self.d4(x)
        return self.output(x)
