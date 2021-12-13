import torch.nn as nn
import torch


class CycleResidualBlock(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, in_channels, out_channels=None):
        super(CycleResidualBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels
        self.rp = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3)
        self.in2 = nn.InstanceNorm2d(out_channels)

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


# noinspection PyTypeChecker
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

    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()

        self.block1 = nn.Sequential(
            # c7s1-64
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_dim, 64, 3),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),
        )

        self.block2 = nn.Sequential(
            # d128
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
        )

        self.block3 = nn.Sequential(
            # d256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
        )

        self.up_block1 = nn.Sequential(
            # u128
            nn.Conv2d(256, 512, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.GroupNorm(32, 128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up_block2 = nn.Sequential(
            # u64
            nn.Conv2d(128, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.last_conv = nn.Conv2d(64, output_dim, 1)
        self.act = nn.Tanh()

        self.model = nn.Sequential(
            # c7s1-64
            nn.ReflectionPad2d(1),
            nn.Conv2d(input_dim, 64, 3),
            nn.GroupNorm(32, 64),
            nn.ReLU(inplace=True),

            # d128
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),

            # d256
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            # R256 x 9
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),
            CycleResidualBlock(256),

            # u128
            nn.Conv2d(256, 512, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.GroupNorm(32, 128),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),

            # u64
            nn.Conv2d(128, 256, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.PixelShuffle(upscale_factor=2),
            nn.GroupNorm(32, 64),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),

            # c7s1-64
            nn.Conv2d(64, output_dim, 1),
            nn.Tanh()
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)

        res_blocks = self.res_blocks(block3)

        x = res_blocks + block3
        up_block1 = self.up_block1(x)
        # x = up_block1 + block2
        up_block2 = self.up_block2(up_block1)
        x = up_block2 + block1

        x = self.act(self.last_conv(x))

        return x


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

    # noinspection PyTypeChecker
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # C64
            nn.Conv2d(input_dim, 64, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            SimpleSelfAttention(64),

            # C128
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SimpleSelfAttention(128),

            # C256
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SimpleSelfAttention(256),

            # C512
            nn.Conv2d(256, 512, 4, padding=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # Output 1-channel prediction map
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x):
        return self.model(x)


class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, bias=True):
        super(MyConvo2d, self).__init__()
        self.padding = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, padding=self.padding, bias=bias)

    def forward(self, x):
        output = self.conv(x)
        return output


class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, x):
        output = self.conv(x)
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2,
                                                                                               1::2]) / 4
        return output


class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, x):
        output = x
        output = (output[:, :, ::2, ::2] + output[:, :, 1::2, ::2] + output[:, :, ::2, 1::2] + output[:, :, 1::2,
                                                                                               1::2]) / 4
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hw):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.ln1 = nn.LayerNorm([input_dim, hw, hw])
        self.ln2 = nn.LayerNorm([input_dim, hw, hw])
        # self.in1 = nn.InstanceNorm2d(input_dim)
        # self.in2 = nn.InstanceNorm2d(input_dim)

        self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size=1)
        self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size=kernel_size, bias=False)
        self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size=kernel_size)

    def forward(self, x):
        shortcut = self.conv_shortcut(x)

        output = x
        output = self.ln1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.ln2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output


# adapted from https://github.com/jalola/improved-wgan-pytorch
class Critic(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, dim=64, output_dim=3):
        super(Critic, self).__init__()

        self.dim = dim
        self.ssize = dim // 16
        # self.ssize = self.dim // 64
        self.output_dim = output_dim

        # out dim
        self.conv1 = MyConvo2d(self.output_dim, self.dim, 3)

        self.rb1 = ResidualBlock(self.dim, 2 * self.dim, 3, self.dim)
        self.rb2 = ResidualBlock(2 * self.dim, 4 * self.dim, 3, self.dim // 2)
        self.attn1 = SelfAttention(4 * self.dim)
        self.rb3 = ResidualBlock(4 * self.dim, 8 * self.dim, 3, self.dim // 4)
        self.attn2 = SelfAttention(8 * self.dim)
        self.rb4 = ResidualBlock(8 * self.dim, 8 * self.dim, 3, self.dim // 8)
        # self.rb5 = ResidualBlock(16 * self.dim, 16 * self.dim, 3)
        # self.rb6 = ResidualBlock(16 * self.dim, 16 * self.dim, 3)

        self.ln1 = nn.Linear(self.ssize * self.ssize * 8 * self.dim, 1)

    def forward(self, x):
        output = x.contiguous()
        # out dim
        output = output.view(-1, self.output_dim, self.dim, self.dim)
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rb2(output)
        output = self.attn1(output)
        output = self.rb3(output)
        output = self.attn2(output)
        output = self.rb4(output)
        output = output.view(-1, self.ssize * self.ssize * 8 * self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output


# adapted from the SAGAN paper
# noinspection PyTypeChecker
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()

        self.ch = in_channels

        # Construct the conv layers
        self.theta = nn.Conv2d(in_channels=self.ch, out_channels=self.ch // 8, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=self.ch, out_channels=self.ch // 8, kernel_size=1)
        self.g = nn.Conv2d(in_channels=self.ch, out_channels=self.ch // 2, kernel_size=1)
        self.o = nn.Conv2d(in_channels=self.ch // 2, out_channels=self.ch, kernel_size=1)

        # Initialize gamma as 0
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # use pooling if attention too compute intensive
        pool = nn.MaxPool2d(kernel_size=2)

        theta = self.theta(x)
        phi = pool(self.phi(x))
        g = pool(self.g(x))
        # phi = self.phi(x)
        # g = self.g(x)

        # Perform reshapes
        theta = theta.view(-1, self.ch // 8, x.shape[2] * x.shape[3])

        # if pooling                                             // 4
        phi = phi.view(-1, self.ch // 8, x.shape[2] * x.shape[3] // 4)

        # if pooling                                         // 4
        g = g.view(-1, self.ch // 2, x.shape[2] * x.shape[3] // 4)

        # Matmul and softmax to get attention maps
        beta = self.softmax(torch.bmm(theta.transpose(1, 2), phi))

        # Attention map times g path
        o = self.o(torch.bmm(g, beta.transpose(1, 2)).view(-1, self.ch // 2, x.shape[2], x.shape[3]))
        return self.gamma * o + x


# adapted from https://github.com/sdoria/SimpleSelfAttention
class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in: int, ks=1):  # , n_out:int):
        super().__init__()
        self.conv = nn.Conv1d(n_in, n_in, ks, padding=ks // 2, bias=False)
        self.gamma = nn.Parameter(torch.Tensor([0.]))
        self.n_in = n_in

    def forward(self, x):
        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)

        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(x, x.permute(0, 2, 1).contiguous())  # (C,N) * (N,C) = (C,C)   => O(NC^2)

        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)

        o = self.gamma * o + x

        return o.view(*size).contiguous()


# noinspection PyTypeChecker
class ColorNet(nn.Module):
    def __init__(self, output_dim=1):
        super(ColorNet, self).__init__()

        encoder = torch.hub.load('RF5/danbooru-pretrained', 'resnet50')
        encoder_layers = list(encoder.children())[0]

        self.block1 = nn.Sequential(*encoder_layers[:3])
        self.block2 = nn.Sequential(*encoder_layers[3:5])
        self.block3 = encoder_layers[5]
        self.block4 = encoder_layers[6]
        self.block5 = encoder_layers[7]

        self.unet_block1 = ColorUnetBlock(2048)
        self.unet_block2 = ColorUnetBlock(1024 + 1024, 512)
        self.unet_block3 = ColorUnetBlock(512 + 512, 256)
        self.attn1 = SelfAttention(256)
        self.unet_block4 = ColorUnetBlock(256 + 256, 64)
        self.attn2 = SelfAttention(64)
        self.unet_block5 = ColorUnetBlock(64 + 64, 64)
        self.attn3 = SimpleSelfAttention(64)

        self.last = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(64, output_dim, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)

        x = self.unet_block1(block5, block_to_concat=None)
        x = self.unet_block2(x, block4)
        x = self.unet_block3(x, block3)
        x = self.attn1(x)
        x = self.unet_block4(x, block2)
        x = self.attn2(x)
        x = self.unet_block5(x, block1)
        x = self.attn3(x)
        x = self.last(x)

        return x


# noinspection PyTypeChecker
class ColorUnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, stride=1, padding=1):
        super(ColorUnetBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels // 2

        self.shuf_conv = nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, stride=stride, padding=0)
        self.act = nn.LeakyReLU(0.2)
        self.upsample = nn.PixelShuffle(upscale_factor=2)
        self.norm = nn.InstanceNorm2d(in_channels // 2, affine=True)

        self.conv1 = nn.Conv2d(in_channels // 2, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, block_to_concat):

        if block_to_concat is not None:
            x = torch.cat([x, block_to_concat], dim=1)

        x = self.shuf_conv(x)
        x = self.act(x)
        x = self.upsample(x)
        x = self.norm(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)

        return x


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
