import torch
from torch import nn
from torchvision.models import vgg16


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.feature_extractor = vgg16().features
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*14*14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8)# 4位置预测 + 4类别预测
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    net = Net()
    print(net)
    input = torch.rand(1, 3, 448, 448)
    output = net(input)# (1, 512, 14, 14)
    print(output)
    print(output.shape)
