import torch.nn as nn


class SimpleNetRGB(nn.Module):
    def __init__(self, num_dates):
        super().__init__()
        input_size = 3*num_dates
        self.conv_1 = nn.Conv2d(input_size, 64, 3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(64, 9, 1)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.relu_1(self.conv_1(x))
        x = self.relu_2(self.conv_2(x))
        x = self.relu_3(self.conv_3(x))
        x = self.global_pool(x)
        x = self.fc_conv(x)
        x = x.squeeze()
        # print(x.shape)
        return x


class SimpleNetFC(nn.Module):
    def __init__(self, num_dates):
        super().__init__()
        input_size = num_dates
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_1 = nn.Linear(3*input_size, 40)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(40, 40)
        self.relu_2 = nn.ReLU(inplace=True)
        self.fc_3 = nn.Linear(40, 40)
        self.relu_3 = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(40, 9)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])

        x = self.global_pool(x)
        x = x.squeeze()
        # print(x.shape)
        x = self.relu_1(self.fc_1(x))
        x = self.relu_2(self.fc_2(x))
        x = self.relu_3(self.fc_3(x))
        x = self.fc_out(x)

        # print(x.shape)
        return x
