import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import cfg

channels_num = 32
encoder_feature_map_channels = [0, 4 * channels_num, 2 * channels_num]
decoder_feature_map_channels = [8 * channels_num, 4 * channels_num, 2 * channels_num]


class Residual(nn.Module):
    def __init__(self, in_dim):
        super(Residual, self).__init__()
        temp_channels = in_dim // 4
        self.conv = nn.Sequential(nn.Conv2d(in_dim, temp_channels, kernel_size=1),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(temp_channels, temp_channels, kernel_size=3, padding=1),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(temp_channels, in_dim, kernel_size=1))
        self.bn = nn.BatchNorm2d(in_dim)

    def forward(self, x):
        out = self.conv(x) + x
        return F.leaky_relu(self.bn(out))


class ResNet(nn.Module):
    def __init__(self, in_dim):
        super(ResNet, self).__init__()
        self.layer = nn.Sequential(Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim),
                                   Residual(in_dim))

    def forward(self, x):
        return self.layer(x)


def conv_bn_relu(in_channels, out_channels):
    blk = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU())
    return blk


# def dilated_conv(in_dim, padding=2, dilation=2):
#     blk = nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, padding=padding, dilation=dilation),
#                         nn.BatchNorm2d(in_dim),
#                         nn.LeakyReLU())
#     return blk


class EncoderNet(nn.Module):
    def __init__(self, in_dim):
        super(EncoderNet, self).__init__()
        layer1, layer2, layer3, layer4 = [], [], [], []

        layer1.append(conv_bn_relu(in_dim, channels_num))
        layer1.append(conv_bn_relu(channels_num, channels_num))

        layer2.append(nn.Conv2d(channels_num, 2 * channels_num, kernel_size=3, stride=2, padding=1))
        layer2.append(nn.LeakyReLU())
        layer2.append(conv_bn_relu(2 * channels_num, 2 * channels_num))
        layer2.append(conv_bn_relu(2 * channels_num, 2 * channels_num))

        layer3.append(nn.Conv2d(2 * channels_num, 4 * channels_num, kernel_size=3, stride=2, padding=1))
        layer3.append(nn.LeakyReLU())
        layer3.append(conv_bn_relu(4 * channels_num, 4 * channels_num))
        layer3.append(conv_bn_relu(4 * channels_num, 4 * channels_num))

        layer4.append(nn.Conv2d(4 * channels_num, 8 * channels_num, kernel_size=3, stride=2, padding=1))
        layer4.append(nn.LeakyReLU())
        layer4.append(conv_bn_relu(8 * channels_num, 8 * channels_num))
        layer4.append(conv_bn_relu(8 * channels_num, 8 * channels_num))

        self.l1 = nn.Sequential(*layer1)
        self.l2 = nn.Sequential(*layer2)
        self.l3 = nn.Sequential(*layer3)
        self.l4 = nn.Sequential(*layer4)

    def forward(self, x, get_feature_map=False):
        out = self.l1(x)
        out = self.l2(out)
        f1 = out
        out = self.l3(out)
        f2 = out
        out = self.l4(out)
        if get_feature_map:
            return out, [f2, f1]
        else:
            return out


class DecoderNet(nn.Module):
    def __init__(self, in_dim, feature_map_channels=None):
        super(DecoderNet, self).__init__()

        f1, f2, f3 = 0, 0, 0
        if feature_map_channels:
            f1, f2, f3 = feature_map_channels

        cat_channels = in_dim + f1
        self.conv1 = nn.Sequential(conv_bn_relu(cat_channels, 8 * channels_num),
                                   conv_bn_relu(8 * channels_num, 8 * channels_num))
        self.deconv1 = nn.Sequential(nn.ConvTranspose2d(8 * channels_num, 4 * channels_num,
                                                        kernel_size=3, stride=2, padding=1,
                                                        output_padding=1),
                                     nn.LeakyReLU())

        cat_channels = 4 * channels_num + f2
        self.conv2 = nn.Sequential(conv_bn_relu(cat_channels, 4 * channels_num),
                                   conv_bn_relu(4 * channels_num, 4 * channels_num))
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(4 * channels_num, 2 * channels_num,
                                                        kernel_size=3, stride=2, padding=1,
                                                        output_padding=1),
                                     nn.LeakyReLU())

        cat_channels = 2 * channels_num + f3
        self.conv3 = nn.Sequential(conv_bn_relu(cat_channels, 2 * channels_num),
                                   conv_bn_relu(2 * channels_num, 2 * channels_num))
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(2 * channels_num, channels_num,
                                                        kernel_size=3, stride=2, padding=1,
                                                        output_padding=1),
                                     nn.LeakyReLU())
        self.conv4 = nn.Sequential(conv_bn_relu(channels_num, channels_num),
                                   conv_bn_relu(channels_num, channels_num))

    def forward(self, x, fuse=None, get_feature_map=False):
        if fuse and fuse[0] is not None:
            x = torch.cat([x, fuse[0]], dim=1)
        out = self.conv1(x)
        f1 = out

        out = self.deconv1(out)
        if fuse and fuse[1] is not None:
            out = torch.cat([out, fuse[1]], dim=1)
        out = self.conv2(out)
        f2 = out

        out = self.deconv2(out)
        if fuse and fuse[2] is not None:
            out = torch.cat([out, fuse[2]], dim=1)
        out = self.conv3(out)
        f3 = out

        out = self.deconv3(out)
        out = self.conv4(out)

        if get_feature_map:
            return out, [f1, f2, f3]
        else:
            return out


class TextConversionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(TextConversionNet, self).__init__()
        self.t_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(8 * channels_num))
        self.s_encoder = nn.Sequential(EncoderNet(in_dim),
                                       ResNet(8 * channels_num))
        self.sk_decoder = nn.Sequential(DecoderNet(16 * channels_num),
                                        nn.Conv2d(channels_num, 1, kernel_size=3, padding=1),
                                        nn.Sigmoid())
        self.t_decoder = DecoderNet(16 * channels_num)
        self.t_conv = conv_bn_relu(channels_num + 1, channels_num + 1)
        self.last = nn.Sequential(nn.Conv2d(channels_num + 1, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x_t, x_s):
        out_t = self.t_encoder(x_t)
        out_s = self.s_encoder(x_s)

        out = torch.cat([out_t, out_s], dim=1)

        out_sk = self.sk_decoder(out)
        out_t = self.t_decoder(out)

        out_t = torch.cat([out_sk, out_t], dim=1)

        out_t = self.t_conv(out_t)

        return out_sk, self.last(out_t)


class BackgroundInpaintingNet(nn.Module):
    def __init__(self, in_dim=3):
        super(BackgroundInpaintingNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(8 * channels_num)
        # self.dilation_net = nn.Sequential(dilated_conv(8 * channels_num),
        #                                   dilated_conv(8 * channels_num, padding=4, dilation=4),
        #                                   dilated_conv(8 * channels_num, padding=8, dilation=8))
        self.decoder = DecoderNet(8 * channels_num, encoder_feature_map_channels)
        self.last = nn.Sequential(nn.Conv2d(channels_num, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x):
        out, f_encoder = self.encoder(x, get_feature_map=True)
        out = self.resnet(out)
        # out = self.dilation_net(out)
        out, fuse = self.decoder(out, [None] + f_encoder, get_feature_map=True)
        return self.last(out), fuse


class FusionNet(nn.Module):
    def __init__(self, in_dim=3):
        super(FusionNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(8 * channels_num)
        self.decoder = DecoderNet(8 * channels_num, decoder_feature_map_channels)
        self.last = nn.Sequential(nn.Conv2d(channels_num, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, x, fuse):
        out = self.encoder(x)
        out = self.resnet(out)
        out = self.decoder(out, fuse)
        return self.last(out)


class NewFusionNet(nn.Module):
    def __init__(self, in_dim=6):
        super(NewFusionNet, self).__init__()
        self.encoder = EncoderNet(in_dim)
        self.resnet = ResNet(8 * channels_num)
        self.decoder = DecoderNet(8 * channels_num)
        self.last = nn.Sequential(nn.Conv2d(channels_num, 3, kernel_size=3, padding=1),
                                  nn.Tanh())

    def forward(self, t_t, t_b):
        out = torch.cat([t_t, t_b], dim=1)
        out = self.encoder(out)
        out = self.resnet(out)
        out = self.decoder(out)
        return self.last(out)


class Generator(nn.Module):
    def __init__(self, in_dim=3):
        super(Generator, self).__init__()
        self.text_conversion_net = TextConversionNet(in_dim)
        self.background_inpainting_net = BackgroundInpaintingNet(in_dim)
        self.fusion_net = FusionNet(in_dim)

    def forward(self, inputs):
        i_t, i_s = inputs
        o_sk, o_t = self.text_conversion_net(i_t, i_s)
        o_b, fuse = self.background_inpainting_net(i_s)
        o_f = self.fusion_net(o_t, fuse)
        return o_sk, o_t, o_b, o_f


class Discriminator(nn.Module):
    def __init__(self, in_dim=6):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=3, stride=2, padding=1),
                                   nn.LeakyReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(128),
                                   nn.LeakyReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(256),
                                   nn.LeakyReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                                   nn.BatchNorm2d(512),
                                   nn.LeakyReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1),
                                   nn.Sigmoid())

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        return self.conv5(out)


class DiscriminatorMixed(nn.Module):
    def __init__(self):
        super(DiscriminatorMixed, self).__init__()
        self.D_background_inpainting = Discriminator()
        self.D_fusion = Discriminator()

    def forward(self, inputs):
        i_db, i_df = inputs
        o_db = self.D_background_inpainting(i_db)
        o_df = self.D_fusion(i_df)
        return o_db, o_df


def get_vgg_model():
    vgg_model = torchvision.models.vgg19()
    pre = torch.load(cfg.vgg19_weights)
    vgg_model.load_state_dict(pre)
    net_list = []
    vgg_layers = [1, 6, 11, 20, 29]
    for i in range(max(vgg_layers) + 1):
        net_list.append(vgg_model.features[i])
    net = torch.nn.Sequential(*net_list)
    return net
