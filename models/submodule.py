import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


def conv2d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    )
    return conv


def conv3d_bn(in_channels, out_channels, kernel_size, stride, padding):
    conv = nn.Sequential(
        nn.BatchNorm3d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
    )
    return conv


def make_conv3d_block(in_channels, hidden_channels, out_channels, num_layers):
    conv3d_block = [conv3d_bn(in_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
    for _ in range(num_layers - 2):
        conv3d_block += [conv3d_bn(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=1)]
    conv3d_block += [conv3d_bn(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)]
    return nn.Sequential(*conv3d_block)


class feature_extraction(nn.Module):
    def __init__(self):
        super(feature_extraction, self).__init__()
        # H x W -> H/2 x W/2
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H/2 x W/2 -> H/4 x W/4
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H/4 x W/4 -> H/8 x W/8
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # H/8 x W/8 -> H/16 x W/16
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # H/16 x W/16 -> H/8 x W/8
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # H/8 x W/8 -> H/4 x W/4
        self.upconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # H/4 x W/4 -> H/2 x W/2
        self.upconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # H/2 x W/2 -> H x W
        self.upconv6 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, output_padding=1)
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()
        )
           





    def forward(self, x):
        # H x W -> H/4 x W/4
        conv1 = self.conv1(x)
        # H/4 x W/4 -> H/8 x W/8
        conv2 = self.conv2(conv1)
        # H/8 x W/8 -> H/16 x W/16
        conv3 = self.conv3(conv2)

        # H/16 x W/16 -> H/8 x W/8
        upconv3 = F.interpolate(input=conv3, scale_factor=2.0, mode='bilinear', align_corners=True)
        upconv3 = torch.nn.functional.interpolate(upconv3, size=conv2.shape[2:], mode='bilinear', align_corners=False)
        res = torch.cat([conv2, upconv3], dim=1)
        res=res[:,:64,:,:]
        conv4 = self.conv4(res)
        # H/8 x W/8 -> H/4 x W/4
        upconv4 = F.interpolate(input=conv4, scale_factor=2.0, mode='bilinear', align_corners=True)
        upconv4 = torch.nn.functional.interpolate(upconv4, size=conv1.shape[2:], mode='bilinear', align_corners=False)
        res1 = torch.cat([conv1, upconv4], dim=1)
        res1 = res1[:,:64,:,:]
        conv5 = self.conv5(res1)
        return [conv3, conv4, conv5]  # [1/16, 1/8, 1/4]
        

class disparityregression(nn.Module):
    def __init__(self, start, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = torch.arange(start=start, end=maxdisp, device='cuda', requires_grad=False).view(1, -1, 1, 1)

    def forward(self, x):
        out = torch.sum(x * self.disp, dim=1, keepdim=True)
        return out


if __name__ == '__main__':
    device = 'cuda'
    summary(feature_extraction().to(device), (3, 256, 256))
