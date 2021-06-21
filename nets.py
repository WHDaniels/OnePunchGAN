import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.rp = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(channels, channels, (3, 3))
        self.in1 = nn.InstanceNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, (3, 3))
        self.in2 = nn.InstanceNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.rp(x)
        out = self.conv1(out)
        out = self.in1(out)
        out = self.relu(out)
        out = self.rp(out)
        out = self.conv2(out)
        out = self.in2(out)
        out += residual
        return self.relu(out)


class Generator(nn.Module):
    """
    CycleGAN generator architecture

    c7s1-64 --> 7x7 Convolution-InstanceNorm_ReLU layer, 64 filters, stride 1
    d128 --> 3x3 Convolution-InstanceNorm_ReLU layer, 128 filters, stride 2, reflection padding
    d256 --> 3x3 Convolution-InstanceNorm_ReLU layer, 256 filters, stride 2, reflection padding
    R256 --> Residual block, two 3x3 convolutional layers, same number of filters on each layer (256)
    R256 ...
    R256 ...
    R256 ...
    R256 ...
    R256 ...
    R256 ...
    R256 ...
    R256 ...
    u128 --> 3x3 fractional-strided-Convolutional-InstanceNorm_ReLU layer, 128 filters, stride 1/2
    u64 --> 3x3 fractional-strided-Convolutional-InstanceNorm_ReLU layer, 64 filters, stride 1/2
    c7s1-3 --> 7x7 Convolution-InstanceNorm_ReLU layer, 3 filters, stride 1
    """

    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(

            # c7s1-64
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, (7, 7)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # d128
            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            # d256
            nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            # R256 x 9
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),
            ResidualBlock(256),

            # u128
            nn.ConvTranspose2d(256, 128, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            # u64
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            # c7s1-64
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, (7, 7)),
            nn.Tanh()

        )

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    """
    CycleGAN discriminator architecture (70x70 PatchGAN)

    (LeakyReLUs are used with a slope of 0.2)

    C64 --> 4x4 Convolution-LeakyReLU layer, 64 filters, stride 2
    C128 --> 4x4 Convolution-InstanceNorm-LeakyReLU layer, 128 filters, stride 2
    C256 --> 4x4 Convolution-InstanceNorm-LeakyReLU layer, 256 filters, stride 2
    C512 --> 4x4 Convolution-InstanceNorm-LeakyReLU layer, 512 filters, stride 2
    Final layer -->  apply a convolution to produce a 1-dimensional output
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(

            # C64
            nn.Conv2d(3, 64, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LeakyReLU(0.2, inplace=True),

            # C128
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # C256
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=(1, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # C512
            nn.Conv2d(256, 512, (4, 4), padding=(1, 1)),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output 1-channel prediction map
            nn.Conv2d(512, 1, (4, 4), padding=(1, 1))

        )

    def forward(self, x):
        return self.model(x)
