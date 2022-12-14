'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn

# Taken from https://github.com/kuangliu/pytorch-cifar

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG11_quad': [64, 'M', 512, 'M', 1024, 1024, 'M', 2048, 2048, 'M', 2048, 512, 'M'],
    'VGG11_doub': [64, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 512, 'M'],
    'VGG11_half': [64, 'M', 64, 'M', 128, 128, 'M', 256, 256, 'M', 256, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, batch_norm=True, bias=True, relu_inplace=True):
        super(VGG, self).__init__()
        self.batch_norm= batch_norm
        self.bias = bias
        self.features = self._make_layers(cfg[vgg_name], relu_inplace=relu_inplace)
        self.classifier = nn.Linear(512, num_classes, bias=self.bias)
        print("Relu Inplace is ", relu_inplace)

    def forward(self, x):
        out = self.features(x)
        # out = out.view(out.size(0), -1)
        out = out.view(out.size(0), out.size(1), -1)
        out = out.mean(axis=2)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg, relu_inplace=True):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if self.batch_norm:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=relu_inplace)]
                else:
                    layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1, bias=self.bias),
                           nn.ReLU(inplace=relu_inplace)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        print("in _make_layers", list(layers))
        return nn.Sequential(*layers)


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
