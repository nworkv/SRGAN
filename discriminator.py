import torch.nn as nn 

class DiscLayer(nn.Module):
  def __init__(self,
               in_channels: int, out_channels: int, stride: int):
    super(DiscLayer, self).__init__()

    self.conv = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                              kernel_size = 3, stride = stride, padding = 1)
    self.norm = nn.BatchNorm2d(out_channels) #nn.InstanceNorm2d(out_channels)
    self.act = nn.LeakyReLU(0.2)

  def forward(self, input):
    input = self.conv(input)
    input = self.norm (input)
    input = self.act (input)
    return input


class Discriminator(nn.Module):
  def  __init__(self,
                in_channels = 3, hid_channels = 64):
    super(Discriminator, self).__init__()
    self.conv = self.conv = nn.Conv2d(in_channels = in_channels, out_channels = hid_channels,
                              kernel_size = 3, stride = 1, padding = 1)
    self.act =  nn.LeakyReLU(0.1)
    self.basic = nn.Sequential(DiscLayer(in_channels = hid_channels, out_channels = hid_channels, stride = 2),
                               DiscLayer(in_channels = hid_channels, out_channels = hid_channels * 2, stride = 1),
                               DiscLayer(in_channels = hid_channels * 2, out_channels = hid_channels * 2, stride = 2),
                               DiscLayer(in_channels = hid_channels * 2, out_channels = hid_channels * 4, stride = 1),
                               DiscLayer(in_channels = hid_channels * 4, out_channels = hid_channels * 4, stride = 2),
                               DiscLayer(in_channels = hid_channels * 4, out_channels = hid_channels * 8, stride = 1),
                               DiscLayer(in_channels = hid_channels * 8, out_channels = hid_channels * 8, stride = 2))

    self.flatten = nn.Flatten()
    self.fc1 = nn.Linear(in_features = 131072, out_features = 1024)

    self.act1 =  nn.LeakyReLU(0.2)
    self.fc2 = nn.Linear(in_features = 1024, out_features = 1)
    self.act2 = nn.Sigmoid()

  def forward(self, input):

    input = self.conv(input)
    input = self.act(input)
    input = self.basic(input)
    input = self.flatten(input)
    input = self.fc1(input)
    input = self.act1(input)
    input = self.fc2(input)
    input = self.act2(input)

    return input