import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3),
                  nn.InstanceNorm2d(dim),
                  nn.ReLU(inplace=True)]
        if use_dropout:
            layers.append(nn.Dropout(0.5))

        layers += [nn.ReflectionPad2d(1),
                   nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3),
                   nn.InstanceNorm2d(dim)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x) + x
        return out


# The Generator class
# The Generator consists of down-sampling layers, Resnet block, and up-sampling layers
class Generator(nn.Module):

    def __init__(self, in_channels, out_channels, out_features = 64, num_res_blocks = 9, use_dropout = False):
        '''
        Parameters:
        in_channels: number of channels in input images
        out_channels: number of channels in output images
        out_features: number of channels after the first convolutional layer
        num_res_blocks: number of ResNet blocks
        use_dropput: whether to use dropout in the resnet block
        '''
        super(Generator, self).__init__()

        layers = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels = in_channels, out_channels = out_features, kernel_size = 7),
            nn.InstanceNorm2d(out_features),
            nn.ReLU(inplace = True)
        ]

        # Downsampling layers, there are 2 down-sampling layers by default
        num_dowsampling = 2 #number of downsampling layers
        curr_channels = out_features
        for i in range(num_dowsampling):
            next_channels = curr_channels * 2
            layers.append(nn.Conv2d(in_channels = curr_channels, out_channels = next_channels, kernel_size = 3, stride = 2, padding = 1))
            layers.append(nn.InstanceNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            curr_channels = next_channels

        # Adding the resnet blocks
        for i in range(num_res_blocks):
            layers.append(ResnetBlock(curr_channels, use_dropout = use_dropout))

        # Upsampling layers, there are 2 up-sampling layers by default
        for i in range(num_dowsampling):
            next_channels = curr_channels // 2
            layers.append(nn.ConvTranspose2d(in_channels=curr_channels, out_channels=next_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
            layers.append(nn.InstanceNorm2d(next_channels))
            layers.append(nn.ReLU(inplace=True))
            curr_channels = next_channels

        layers += [nn.ReflectionPad2d(3),
                   nn.Conv2d(in_channels = out_features, out_channels = out_channels, kernel_size = 7),
                   nn.Tanh()]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# The Discriminator class
class Discriminator(nn.Module):
    def __init__(self, in_channels, out_features = 64):
        '''
        Parameters
        in_channels: number of channels in input images
        out_features: number of channels after the first convolutional layer
        '''
        super(Discriminator, self).__init__()
        layers = []

        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_features, kernel_size=4, stride=2, padding=1),
                   nn.LeakyReLU(0.2, inplace=True)]

        curr_channels = out_features
        num_layers = 3 # number of convolutional layers

        for i in range(1, num_layers):
            next_channels = curr_channels * 2
            layers += [nn.Conv2d(in_channels= curr_channels, out_channels=next_channels, kernel_size=4, stride=2, padding=1),
                       nn.InstanceNorm2d(next_channels),
                       nn.LeakyReLU(0.2, inplace=True)]
            curr_channels = next_channels

        next_channels = curr_channels * 2
        layers += [nn.Conv2d(in_channels=curr_channels, out_channels=next_channels, kernel_size=4, stride=1, padding=1),
                   nn.InstanceNorm2d(next_channels),
                   nn.LeakyReLU(0.2, inplace=True)]
        curr_channels = next_channels

        # output one channel prediction map
        layers += [nn.Conv2d(in_channels= curr_channels, out_channels=1, kernel_size=4, stride=1, padding=1)]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# The building block of U-Net
# Two consecutive convolutional layers

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# The building block of U-Net
# Down-sampling layers of U-Net

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# The building block of U-Net
# Up-sampling layers of U-Net

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x = torch.cat((x2, x1), dim=1)
        x = self.conv2(x)
        return x

class U_net(nn.Module):
    def __init__(self, in_channels, out_channels, out_features = 64):
        super(U_net, self).__init__()
        self.conv1 = DoubleConv(in_channels, out_features)
        self.conv2 = Down(out_features, out_features * 2)
        self.conv3 = Down(out_features * 2, out_features * 4)
        self.conv4 = Down(out_features * 4, out_features * 8)
        self.conv5 = Down(out_features * 8, out_features * 16)
        self.deconv4 = Up(out_features * 16, out_features * 8)
        self.deconv3 = Up(out_features * 8, out_features * 4)
        self.deconv2 = Up(out_features * 4, out_features * 2)
        self.deconv1 = Up(out_features * 2, out_features)
        layers=  [
            nn.Conv2d(in_channels=out_features, out_channels=out_channels, kernel_size=3, padding=1),
            nn.Tanh()
        ]
        self.output = nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        out = self.deconv4(x5, x4)
        out = self.deconv3(out, x3)
        out = self.deconv2(out, x2)
        out = self.deconv1(out, x1)
        out = self.output(out)
        return out



