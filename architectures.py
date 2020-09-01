from collections import OrderedDict

import torch
from torch import nn


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 64, 9, padding=4)
        self.layer2 = nn.Conv2d(64, 32, 5, padding=2)
        self.layer3 = nn.Conv2d(32, 3, 3, padding=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.layer1(image)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


class ESPCNN(nn.Module):
    def __init__(self):
        super(ESPCNN, self).__init__()
        self.layer1 = nn.Conv2d(3, 128, 5, padding=2)
        self.layer2 = nn.Conv2d(128, 64, 5, padding=2)
        self.layer3 = nn.ConvTranspose2d(
            64, 3, 3, stride=2, padding=1, output_padding=1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.layer1(image)
        x = self.activation(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x


class EDSR(nn.Module):
    def __init__(self, width=256, depth=32):
        super(EDSR, self).__init__()
        self.input_conv = nn.Conv2d(3, width, 9, padding=4)
        self.main = nn.Sequential(OrderedDict(
            [(f"ResBlock{i}", self.ResBlock(width)) for i in range(depth)]))
        self.conv2 = nn.Conv2d(width, width, 3, padding=1)
        self.conv3 = nn.Conv2d(width, width * 4, 3, padding=1)
        self.conv4 = nn.Conv2d(width, 3, 3, padding=1)
        self.pixel_rearrange = nn.PixelShuffle(2)
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        x = self.activation1(self.input_conv(image))
        sub_x = self.main(x)
        sub_x = self.activation2(self.conv2(sub_x))
        x = x + sub_x
        x = self.activation3(self.conv3(x))
        x = self.pixel_rearrange(x)
        x = self.sigmoid(self.conv4(x))
        return x

    class ResBlock(nn.Module):
        def __init__(self, width):
            super(EDSR.ResBlock, self).__init__()
            self.basic_block = (nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1),
                nn.PReLU(),
                nn.Conv2d(width, width, 3, padding=1),
                nn.PReLU())
            )

        def forward(self, x):
            return 0.01 * self.basic_block(x) + x


class MDSR(nn.Module):
    def __init__(self, width=64, depth=80):
        super(MDSR, self).__init__()
        self.input_conv = nn.Conv2d(1, width, 3, padding=1)
        self.preproc2 = nn.Sequential(self.ResBlock(width, 5),
                                      self.ResBlock(width, 5))
        self.preproc4 = nn.Sequential(self.ResBlock(width, 5),
                                      self.ResBlock(width, 5))
        self.preproc6 = nn.Sequential(self.ResBlock(width, 5),
                                      self.ResBlock(width, 5))
        self.main = nn.Sequential(OrderedDict(
            [(f"ResBlock{i}", self.ResBlock(width)) for i in range(depth)]))
        self.upscale2 = nn.Sequential(
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Sigmoid())
        self.upscale4 = nn.Sequential(
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Sigmoid())
        self.upscale6 = nn.Sequential(
            nn.Conv2d(width, 9 * width, 3, padding=1),
            nn.PixelShuffle(3),
            nn.Conv2d(width, 4 * width, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(width, 1, 3, padding=1),
            nn.Sigmoid())
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.activation3 = nn.PReLU()

    def forward(self, image, scale):
        x = self.activation1(self.input_conv(image))
        if (scale == 2):
            x = self.preproc2(x)
        elif (scale == 4):
            x = self.preproc4(x)
        elif (scale == 6):
            x = self.preproc6(x)
        x = x + self.main(x)
        if (scale == 2):
            x = self.upscale2(x)
        elif (scale == 4):
            x = self.upscale4(x)
        elif (scale == 6):
            x = self.upscale6(x)
        return x

    class ResBlock(nn.Module):
        def __init__(self, width, kernel_size=3):
            super(MDSR.ResBlock, self).__init__()
            self.basic_block = (nn.Sequential(
                nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2),
                nn.PReLU(),
                nn.Conv2d(width, width, kernel_size, padding=kernel_size // 2)
            ))

        def forward(self, x):
            return 0.01 * self.basic_block(x) + x


class RCAN(nn.Module):
    def __init__(self, channel_n, group_n, block_n):
        super(RCAN, self).__init__()
        self.main = nn.Sequential(OrderedDict(
            [(f"Group_{i}", self.ResidualGroup(channel_n, block_n)) for i in range(group_n)]))
        self.conv1 = nn.Conv2d(1, channel_n, 3, padding=1)
        self.conv2 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
        self.conv3 = nn.Conv2d(channel_n, channel_n * 4, 3, padding=1)
        self.conv4 = nn.Conv2d(channel_n, 1, 3, padding=1)
        self.upscale = nn.PixelShuffle(2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        subx = self.main(x)
        subx = self.conv2(subx)
        x = x + subx
        x = self.conv3(x)
        x = self.upscale(x)
        x = self.conv4(x)
        x = self.sigmoid(x)
        return x

    class ResidualGroup(nn.Module):
        def __init__(self, channel_n, block_n):
            super(RCAN.ResidualGroup, self).__init__()
            self.main = nn.Sequential(OrderedDict(
                [(f"RCAB_{i}", self.AttentionBlock(channel_n)) for i in range(block_n)]))
            self.conv = nn.Conv2d(channel_n, channel_n, 3, padding=1)

        def forward(self, x):
            subx = self.main(x)
            subx = self.conv(subx)
            x = x + subx
            return x

        class AttentionBlock(nn.Module):
            def __init__(self, channel_n):
                super(RCAN.ResidualGroup.AttentionBlock, self).__init__()
                self.conv1 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
                self.conv2 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
                self.relu = nn.ReLU()
                self.attention = self.AttentionPart(channel_n, 16)

            def forward(self, x):
                subx = self.conv1(x)
                subx = self.relu(subx)
                subx = self.conv2(subx)
                subx = self.attention(subx)
                x = x + subx
                return x

            class AttentionPart(nn.Module):
                def __init__(self, channel_n, ratio):
                    super(RCAN.ResidualGroup.AttentionBlock.AttentionPart,
                          self).__init__()
                    middle = channel_n // ratio
                    self.glob_pool = nn.AdaptiveAvgPool2d(1)
                    self.conv1 = nn.Conv2d(channel_n, middle, 1)
                    self.conv2 = nn.Conv2d(middle, channel_n, 1)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    attention = self.glob_pool(x)
                    attention = self.conv1(attention)
                    attention = self.relu(attention)
                    attention = self.conv2(attention)
                    attention = self.sigmoid(attention)
                    x = x * attention
                    return x


class MRCAN(nn.Module):
    def __init__(self, channel_n, group_n, block_n):
        super(MRCAN, self).__init__()
        self.main = nn.Sequential(OrderedDict(
            [(f"Group_{i}", self.ResidualGroup(channel_n, block_n)) for i in range(group_n)]))
        self.conv1 = nn.Conv2d(1, channel_n, 3, padding=1)
        self.conv2 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
        self.tailx2 = nn.Sequential(
            nn.Conv2d(channel_n, channel_n * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel_n, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.tailx4 = nn.Sequential(
            nn.Conv2d(channel_n, channel_n * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel_n, channel_n * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel_n, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.tailx6 = nn.Sequential(
            nn.Conv2d(channel_n, channel_n * 9, 3, padding=1),
            nn.PixelShuffle(3),
            nn.Conv2d(channel_n, channel_n * 4, 3, padding=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channel_n, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, scale):
        x = self.conv1(x)
        subx = self.main(x)
        subx = self.conv2(subx)
        x = x + subx
        if (scale == 2):
            x = self.tailx2(x)
        if (scale == 4):
            x = self.tailx4(x)
        if (scale == 6):
            x = self.tailx6(x)
        return x

    class ResidualGroup(nn.Module):
        def __init__(self, channel_n, block_n):
            super(MRCAN.ResidualGroup, self).__init__()
            self.main = nn.Sequential(OrderedDict(
                [(f"RCAB_{i}", self.AttentionBlock(channel_n)) for i in range(block_n)]))
            self.conv = nn.Conv2d(channel_n, channel_n, 3, padding=1)

        def forward(self, x):
            subx = self.main(x)
            subx = self.conv(subx)
            x = x + subx
            return x

        class AttentionBlock(nn.Module):
            def __init__(self, channel_n):
                super(MRCAN.ResidualGroup.AttentionBlock, self).__init__()
                self.conv1 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
                self.conv2 = nn.Conv2d(channel_n, channel_n, 3, padding=1)
                self.relu = nn.ReLU()
                self.attention = self.AttentionPart(channel_n, 16)

            def forward(self, x):
                subx = self.conv1(x)
                subx = self.relu(subx)
                subx = self.conv2(subx)
                subx = self.attention(subx)
                x = x + subx
                return x

            class AttentionPart(nn.Module):
                def __init__(self, channel_n, ratio):
                    super(MRCAN.ResidualGroup.AttentionBlock.AttentionPart,
                          self).__init__()
                    middle = channel_n // ratio
                    self.glob_pool = nn.AdaptiveAvgPool2d(1)
                    self.conv1 = nn.Conv2d(channel_n, middle, 1)
                    self.conv2 = nn.Conv2d(middle, channel_n, 1)
                    self.relu = nn.ReLU()
                    self.sigmoid = nn.Sigmoid()

                def forward(self, x):
                    attention = self.glob_pool(x)
                    attention = self.conv1(attention)
                    attention = self.relu(attention)
                    attention = self.conv2(attention)
                    attention = self.sigmoid(attention)
                    x = x * attention
                    return x


class SharpeningModule(nn.Module):
    def __init__(self, type):
        super(SharpeningModule, self).__init__()
        self.type = type
        if (type == "simple"):
            self.transform_tensor = torch.tensor(
                [[-1., -1., -1.],
                 [-1., 9., -1.],
                 [-1., -1., -1.]])
            self.transform_tensor = self.transform_tensor.view(
                1, 1, 3, 3).to(torch.device("cuda:0"))
        if (type == "addition"):
            self.transform_tensor = torch.tensor(
                [[0., -1, 0.],
                 [-1, 4., -1],
                 [0., -1, 0.]])
            self.transform_tensor = self.transform_tensor.view(
                1, 1, 3, 3).to(torch.device("cuda:0"))
        if (type == "unblurring"):
            self.transform_tensor = torch.tensor(
                [[0.111, 0.111, 0.111],
                 [0.111, 0.111, 0.111],
                 [0.111, 0.111, 0.111]])
            self.transform_tensor = self.transform_tensor.view(
                1, 1, 3, 3).to(torch.device("cuda:0"))

    def forward(self, x):
        x = self.__reversedSigmoid(x)
        red_channel = torch.nn.functional.conv2d(
            x[:, :1], self.transform_tensor, padding=1)
        green_channel = torch.nn.functional.conv2d(
            x[:, 1:2], self.transform_tensor, padding=1)
        blue_channel = torch.nn.functional.conv2d(
            x[:, 2:3], self.transform_tensor, padding=1)
        if (self.type == "simple"):
            x = torch.cat((red_channel, green_channel, blue_channel), dim=1)
        if (self.type == "addition"):
            x = x + torch.cat((red_channel, green_channel,
                               blue_channel), dim=1)
        if (self.type == "unblurring"):
            x = 2 * x - \
                torch.cat((red_channel, green_channel, blue_channel), dim=1)
        x = torch.nn.Sigmoid()(x)
        return x

    def __minMaxScaler(self, x):
        x -= torch.min(x)
        x /= torch.max(x)
        return x

    def __reversedSigmoid(self, x):
        y = torch.log(x / (1 - x))
        return y
