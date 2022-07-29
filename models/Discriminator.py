import torch
import torch.nn as nn
import math


class Discriminator(nn.Module):
    def __init__(self, opt, in_features=3, out_features=64):
        super(Discriminator, self).__init__()

        def block(in_features, out_features):
            layers = [
                nn.Conv2d(in_features, out_features, 3, 2, 1),
                #nn.Conv2d(in_features, out_features, 5, 2, 1),
                #nn.Dropout(p=1 - opt.dropout_prob),
                #nn.BatchNorm2d(out_features),
                #nn.InstanceNorm2d(out_features, affine=True),
                nn.LeakyReLU(negative_slope=opt.lrelu_slope, inplace=True)
            ]

            return layers

        self.l1 = nn.Sequential(
            nn.Conv2d(opt.channels, 16, 3, 1, 1),  #16
            #nn.Dropout(p=1 - opt.dropout_prob),
            #nn.BatchNorm2d(64),
            #nn.InstanceNorm2d(16, affine=True),
            nn.LeakyReLU(negative_slope=opt.lrelu_slope, inplace=True))

        # Use 16,32 for rotating mnist and chisq
        # in some code they use 3 blocs
        self.blocks = nn.Sequential(*block(16, 32), *block(32, 128),
                                    *block(128, 256), *block(256, 512),
                                    *block(512, 1024))

        # self.l2 = nn.Sequential(
        #     nn.Linear(self.blocks[-4].out_channels *\
        #               (int(opt.img_size / 2**(len(self.blocks) / 4)))**2,
        #               1), nn.Sigmoid())
        self.l2 = nn.Sequential(
            #self.blocks[-2] and len(self.blocks)/3 if no InstaceNorm
            nn.Linear(self.blocks[-2].out_channels *\
                      (int(opt.img_size / 2**(len(self.blocks) / 2)))**2,
                      1))#, nn.Sigmoid())
        #self.l2 = nn.Sequential(nn.Linear(7200, 1))
        if opt.gan_type == 'gan':
            self.l2.add_module('1', nn.Sigmoid())
            # self.l2 = nn.Sequential(
            # nn.Linear(self.blocks[-2].out_channels *\
            #           (int(opt.img_size / 2**(len(self.blocks) / 2)))**2,
            #           1), nn.Sigmoid())

    def forward(self, img):
        x = self.l1(img)
        for layer in self.blocks:
            x = layer(x)
        out = self.l2(x.view(img.shape[0], -1))

        return out


class DiscriminatorFConv(nn.Module):
    # Fully convolutional discriminator. No linear layers are presented
    def __init__(self, opt, in_features=3, out_features=64):
        super(DiscriminatorFConv, self).__init__()

        def block(in_features, out_features, k=3, s=1, p=1, bn=True):
            layers = [
                nn.Conv2d(in_features, out_features, k, s, p, bias=False),
                # nn.Dropout(p=1 - opt.dropout_prob)
            ]
            if bn:
                layers.append(nn.BatchNorm2d(out_features))
                #nn.InstanceNorm2d(out_features, affine=True)
            layers.append(
                nn.LeakyReLU(negative_slope=opt.lrelu_slope, inplace=True))

            return layers

        if opt.img_size == 32:
            # size 32
            block_inch = [opt.channels, 64, 128, 256]
            block_outch = [64, 128, 256, 1]
            stride = [2, 2, 2, 1]
            padding = [1, 1, 1, 0]
        elif opt.img_size == 64:
            # size 64
            block_inch = [opt.channels, 64, 64, 128, 256]
            block_outch = [64, 64, 128, 256, 1]
            stride = [2, 2, 2, 2, 1]
            padding = [1, 1, 1, 1, 0]
        elif opt.img_size == 304:
            # size 304
            block_inch = [opt.channels, 4, 8, 16, 32, 64, 128, 256]
            block_outch = [4, 8, 16, 32, 64, 128, 256, 1]
            stride = [2, 2, 2, 2, 2, 2, 2, 2, 1]
            padding = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        else:
            block_inch = [opt.channels, 64, 64, 128, 256]
            block_outch = [64, 64, 128, 256, 1]
            stride = [2, 2, 2, 2, 1]
            padding = [1, 1, 1, 1, 0]

        blocks = []
        bn = opt.gan_type != 'wgan'
        if bn:
            blocklen = 3
        else:
            blocklen = 2
        for i in range(len(block_inch)):
            blocks.extend(
                block(block_inch[i],
                      block_outch[i],
                      4,
                      stride[i],
                      padding[i],
                      bn=bn))

        blocks = blocks[:-(blocklen - 1)]  #last block keep convolution only
        print(blocks)
        self.blocks = nn.Sequential(*blocks)

        if opt.gan_type == 'gan':
            self.blocks.add_module('1', nn.Sigmoid())
        self.add_noise = opt.disc_noise

    def forward(self, img):
        if self.add_noise:
            img = img + torch.rand(img.shape).to(img) * 0.4

        x = img
        for layer in self.blocks:
            x = layer(x)

        out = x.view(-1, 1)
        return out
