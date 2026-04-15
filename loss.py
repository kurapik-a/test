from torch import nn


class FeatureLoss(nn.Module):
    def __init__(self):
        super(FeatureLoss, self).__init__()

        # 位置信息用MSE，类别用交叉熵
        self.location_loss = nn.MSELoss()
        self.class_loss = nn.CrossEntropyLoss()

    # predict 尺寸[batch_size, 8]
    def forward(self, predicts, targets):

        location_loss = self.location_loss(predicts, targets)
