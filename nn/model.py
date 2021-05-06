import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, class_amount, image_size, print_info=False):
        super().__init__()

        self.print_info = print_info
        width = 6

        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(3, width, 5)
        # (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(width, image_size // 2, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear((image_size // 2) * 29 * 29, 120) # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, class_amount)


    def forward(self, x):
        if self.print_info:
            print(x.shape)
        # [Batch size, color channels, image_height, image_width] 
        x = self.pool(F.relu(self.conv1(x)))
        if self.print_info:
            print(x.shape)
        x = self.pool(F.relu(self.conv2(x)))
        if self.print_info:
            print(x.shape)
        # flatten all dimensions except the batch dimension
        x = torch.flatten(x, 1)
        if self.print_info:
            print(x.shape)
        x = F.relu(self.fc1(x))
        if self.print_info:
            print(x.shape)
        x = F.relu(self.fc2(x))
        if self.print_info:
            print(x.shape)
        x = self.fc3(x)
        if self.print_info:
            print(x.shape)
        return x
