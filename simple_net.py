import torch.nn as nn


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SimpleNetRGB(nn.Module):
    def __init__(self, num_dates, channels_in=3):
        super().__init__()
        input_size = channels_in*num_dates
        self.conv_1 = nn.Conv2d(input_size, 64, 3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(128, 9, 1)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.conv_1(x)
        x = self.relu_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)

        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.global_pool(x)
        x = self.fc_conv(x)
        x = x.squeeze()
        # print(x.shape)
        return x


class ModerateNetRGB(nn.Module):
    def __init__(self, num_dates, dropout_rate=0.6):
        super().__init__()
        input_size = 3*num_dates
        self.conv_1 = nn.Conv2d(input_size, 64, 3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.drop_1 = nn.Dropout2d(dropout_rate)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.bn_1 = nn.BatchNorm2d(64)
        self.conv_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.drop_2 = nn.Dropout2d(dropout_rate)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.bn_2 = nn.BatchNorm2d(64)
        self.conv_3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)
        self.drop_3 = nn.Dropout2d(dropout_rate)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(128, 9, 1)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.pool_1(x)

        x = self.bn_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = self.pool_2(x)

        x = self.bn_2(x)
        x = self.conv_3(x)
        x = self.relu_3(x)
        x = self.drop_3(x)
        x = self.global_pool(x)
        x = self.fc_conv(x)
        x = x.squeeze()
        # print(x.shape)
        return x


class SimpleNet3D(nn.Module):
    def __init__(self, num_dates, channels_in=3):
        super().__init__()
        input_size = 3*num_dates
        self.conv_1 = nn.Conv3d(channels_in, 32, 3, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.bn_2 = nn.BatchNorm3d(32)
        self.pool_2 = nn.Conv3d(32, 32, 3, padding=1, stride=2)  # nn.MaxPool3d(2, 2)
        self.conv_2 = nn.Conv3d(32, 64, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.bn_3 = nn.BatchNorm3d(64)
        self.pool_3 = nn.Conv3d(64, 64, 3, padding=1, stride=2)  # nn.MaxPool3d(2, 2)
        self.conv_3 = nn.Conv3d(64, 128, 3, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv_1 = nn.Conv2d(384, 128, 1)
        self.relu_fc_1 = nn.ReLU(inplace=True)
        self.fc_conv_2 = nn.Conv2d(128, 9, 1)
        self.relu_fc_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.contiguous()
        # print(x.shape)

        x = x.permute(0, 2, 1, 3, 4)
        x = self.relu_1(self.conv_1(x))
        x = self.bn_2(x)
        # print(x.shape)

        x = self.pool_2(x)
        x = self.relu_2(self.conv_2(x))
        x = self.bn_3(x)
        # print(x.shape)

        x = self.pool_3(x)
        x = self.relu_3(self.conv_3(x))
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        # print(x.shape)
        x = self.global_pool(x)
        x = self.relu_fc_1(self.fc_conv_1(x))
        x = self.relu_fc_2(self.fc_conv_2(x))
        x = x.squeeze()
        # print(x.shape)
        return x


class SimpleNetAttentionRGB(nn.Module):
    def __init__(self, num_dates):
        super().__init__()
        input_size = 3*num_dates
        self.group_conv = nn.Conv2d(input_size, input_size//3, 3, groups=num_dates, padding=1)
        self.relu_1 = nn.ReLU(inplace=True)
        self.se_time = SELayer(input_size//3, 1)
        self.conv_2 = nn.Conv2d(input_size//3, 64, 3, padding=1)
        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu_3 = nn.ReLU(inplace=True)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(64, 9, 1)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.relu_1(self.group_conv(x))
        x = self.se_time(x)
        x = self.relu_2(self.conv_2(x))
        x = self.relu_3(self.conv_3(x))
        x = self.global_pool(x)
        x = self.fc_conv(x)
        x = x.squeeze()
        # print(x.shape)
        return x


class SimpleNetDeeperRGB(nn.Module):
    def __init__(self, num_dates):
        super().__init__()
        input_size = 3*num_dates
        self.in_conv = nn.Conv2d(input_size, 32, 3, padding=1)

        self.relu_2 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(32, 64, 3, padding=1)

        self.relu_3 = nn.ReLU(inplace=True)
        self.conv_3 = nn.Conv2d(64, 64, 3, padding=1)

        self.relu_4 = nn.ReLU(inplace=True)
        self.conv_4 = nn.Conv2d(64, 64, 3, padding=1)

        self.relu_5 = nn.ReLU(inplace=True)
        self.conv_5 = nn.Conv2d(64, 128, 3, padding=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc_conv = nn.Conv2d(128, 9, 1)

    def forward(self, x):
        x = x.contiguous()
        x = x.view(x.shape[0], -1, x.shape[3], x.shape[4])
        x = self.in_conv(x)
        x = self.conv_2(self.relu_2(x))
        x_shortcut = x
        x = self.conv_3(self.relu_3(x))
        x = self.conv_4(self.relu_4(x))
        x = x + x_shortcut
        x = self.conv_5(self.relu_5(x))

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
