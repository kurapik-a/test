import torch
from torch import nn


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

        # 位置信息用MSE，类别用交叉熵
        self.location_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

    # predict 尺寸[batch_size, 8]
    def forward(self, predicts, targets):

        location_loss = self.location_loss(predicts[:, 0:4], targets[:, 0:4])
        class_loss = self.class_loss(predicts[:, 4:8], targets[:, 4:8])

        return location_loss, class_loss

if __name__ == '__main__':
    loss = FeatureLoss()
    predicts = torch.rand(1,8)
    targets = torch.tensor([
        [0.5, 0.6, 1.2, 0.3, 1.0, 0.0, 0.0, 0.0]
    ])
    print(predicts.shape)
    print(targets.shape)
    print(loss(predicts, targets))