from torch import nn

class ConvLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, max_pool=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                  out_channels=out_channels,
                  kernel_size=kernel_size)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm2d(num_features=out_channels)
        if max_pool:
            self.max_pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.max_pool = None

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.batchnorm(x)

        if self.max_pool:
            x = self.max_pool(x)


        return x.data
