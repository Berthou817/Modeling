import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from collections import OrderedDict
from torch.nn.init import kaiming_normal_

#双重卷积
class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel, 0.1),
            nn.ReLU(True),
            nn.Conv3d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channel, 0.1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.model(x)
# 下采样
class DOWN(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv3d(in_channel, in_channel, kernel_size=2, stride=2),
            DoubleConv(in_channel, out_channel))

    def forward(self, x):
        return self.model(x)
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)
class PixelNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = torch.mul(x, x) # or x ** 2
        tmp1 = torch.rsqrt(torch.mean(tmp, dim=1, keepdim=True) + self.epsilon)

        return x * tmp1
class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        super(FC, self).__init__()
        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(torch.randn(out_channels, in_channels) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            out = F.linear(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul)
        else:
            out = F.linear(x, self.weight * self.w_lrmul)
        out = F.leaky_relu(out, 0.2, inplace=True)
        return out
class G_mapping(nn.Module):
    def __init__(self,
                 mapping_fmaps=256,
                 dlatent_size=256,
                 resolution=256,
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = nn.Sequential(
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
        )

        self.normalize_latents = normalize_latents
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2
        self.pixel_norm = PixelNorm()

    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers
class Conv3d(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1,
                 bias=True):
        super().__init__()
        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)
        self.kernel_size = kernel_size
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        self.weight = torch.nn.Parameter(
            torch.randn(output_channels, input_channels, kernel_size, kernel_size, kernel_size) * init_std)
        if bias:
            self.bias = torch.nn.Parameter(torch.zeros(output_channels))
            self.b_lrmul = lrmul
        else:
            self.bias = None

    def forward(self, x):
        if self.bias is not None:
            return F.conv3d(x, self.weight * self.w_lrmul, self.bias * self.b_lrmul, padding=self.kernel_size // 2)
        else:
            return F.conv3d(x, self.weight * self.w_lrmul, padding=self.kernel_size // 2)
class Upscale3d(nn.Module):
    def __init__(self, factor=2, gain=1):
        super().__init__()
        self.gain = gain
        self.factor = factor


    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = x.view(shape[0], shape[1], shape[2], 1, shape[3], 1, shape[4], 1).expand(-1, -1, -1, self.factor, -1, self.factor, -1, self.factor)
            x = x.contiguous().view(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3], self.factor * shape[4])

        return x
class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):

        if noise is None:
            noise = torch.zeros(x.size(0), 1, x.size(2), x.size(3), x.size(4), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1, 1) * noise.to(x.device)
class ApplyStyle(nn.Module):
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):

        style = self.linear(latent)

        shape = [-1, 2, x.size(1), 1, 1 ,1]

        style = style.view(shape)

        x = x * (style[:, 0] + 1.) + style[:, 1]
        return x
class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp
class LayerEpilogue(nn.Module):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        super(LayerEpilogue, self).__init__()
        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm()
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):

        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)
        return x
class GBlock(nn.Module):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,
                 dlatent_size=256,
                 use_style=True,
                 f=None,
                 factor=2,
                 fmap_base=4096,
                 fmap_decay=1.0,
                 fmap_max=32,
                 ):
        super(GBlock, self).__init__()

        self.nf = 32
        self.res = res


        self.noise_input = noise_input

        self.up_sample = Upscale3d(factor)
        self.adaIn1 = LayerEpilogue(self.nf, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv3d(input_channels=self.nf, output_channels=self.nf,
                             kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf, dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):

        x = self.up_sample(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv1(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x

class G_synthesis(nn.Module):
    def __init__(self,
                 dlatent_size,
                 resolution=256,
                 fmap_base=4096,
                 num_channels=1,
                 structure='fixed',
                 fmap_max=128,
                 fmap_decay=1.0,
                 f=None,
                 use_pixel_norm      = True,
                 use_instance_norm   = True,
                 use_wscale = True,
                 use_noise = True,
                 use_style = True
                 ):

        super(G_synthesis, self).__init__()

        self.nf = [32,32,32,32,32,32,32,32,32,32,16,16,16,16,16]
        self.structure = structure

        self.resolution_log2 = int(np.log2(resolution))
        num_layers = self.resolution_log2 * 2 - 2 # 14
        self.num_layers = num_layers

        self.noise_inputs = []
        for layer_idx in range(num_layers):
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res, 2 ** res] #4, 8, 16, 32,64,128,256
            self.noise_inputs.append(torch.randn(*shape).to("cuda"))


        self.channel_shrinkage = Conv3d(input_channels=32,
                                        output_channels=16,
                                        kernel_size=3,
                                        use_wscale=use_wscale)
        self.torgb = Conv3d(16, num_channels, kernel_size=1, gain=1, use_wscale=use_wscale)

        self.const_input = nn.Parameter(torch.ones(1, self.nf[1] , 4, 4, 4))
        self.bias = nn.Parameter(torch.ones(self.nf[1]))
        self.adaIn1 = LayerEpilogue(self.nf[1], dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv3d(input_channels=self.nf[1], output_channels=self.nf[1], kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf[1], dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)
        self.transform= Conv3d(32,16,1,1,use_wscale=use_wscale)


        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)


        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)


        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)
        res = 8
        self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                      self.noise_inputs)
    def forward(self, dlatent):

        images_out = None

        if self.structure == 'fixed':

            x = self.const_input.expand(dlatent.size(0), -1, -1, -1, -1)
            x = x + self.bias.view(1, -1, 1, 1, 1)
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])
            x = self.GBlock1(x, dlatent)

            x = self.GBlock2(x, dlatent)
            #

            x = self.GBlock3(x, dlatent)
            #

            x = self.GBlock4(x, dlatent)
            x = self.GBlock5(x, dlatent)
            x = self.GBlock6(x, dlatent)



            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out

class StyleGenerator(nn.Module):
    def __init__(self,
                 mapping_fmaps=256,
                 style_mixing_prob=0.9,
                 truncation_psi=0.5,
                 truncation_cutoff=14,
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):

        # latents1 = torch.cat((latents1, pore), 1)



        dlatents1, num_layers = self.mapping(latents1)
        dlatents1 = dlatents1.unsqueeze(1)
        dlatents1 = dlatents1.expand(-1, int(num_layers), -1)



        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi


            dlatents1 = dlatents1 * torch.Tensor(coefs).to(dlatents1.device)


        img = self.synthesis(dlatents1)
        img = F.sigmoid(img)
        return img

class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(nn.InstanceNorm3d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv3d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                                  nn.InstanceNorm3d(in_dim, affine=True),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                  nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                                  nn.AvgPool3d(kernel_size=2, stride=2, padding=0))

        self.short_cut = nn.Sequential(nn.AvgPool3d(kernel_size=2, stride=2, padding=0),
                                       nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1, padding=0))

    def forward(self, x):

        out = self.conv(x)+self.short_cut(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=2, stride=2, padding=1),
            nn.Conv3d(16, 32, kernel_size=2, stride=2, padding=1),
            nn.Conv3d(32, 64, kernel_size=2, stride=2, padding=1)
        )
        self.res_blocks = nn.Sequential(ResBlock(64, 128),
                                        ResBlock(128, 192),
                                        ResBlock(192, 256))
        self.pool_block = nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                        nn.AvgPool3d(kernel_size=2, stride=2, padding=0))

        # Return mu and logvar for reparameterization trick
        self.fc_mu = nn.Linear(2048, 256)
        self.fc_logvar = nn.Linear(2048, 256)

    def forward(self, x):
        # (N, 16, 128, 128) -> (N, 64, 64, 64)
        out = self.conv(x)
        # (N, 64, 64, 64) -> (N, 128, 32, 32) -> (N, 192, 16, 16) -> (N, 256, 8, 8)
        out = self.res_blocks(out)
        # (N, 256, 8, 8) -> (N, 256, 1, 1)
        out = self.pool_block(out)
        # (N, 256, 1, 1) -> (N, 256)
        out = out.view(x.size(0), -1)

        # (N, 256) -> (N, z_dim) x 2
        mu = self.fc_mu(out)
        log_var = self.fc_logvar(out)

        return (mu, log_var)
class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, k=4, s=2, p=1, norm=True, non_linear='leaky_relu'):
        super(ConvBlock, self).__init__()
        layers = []

        # Convolution Layer
        layers += [nn.Conv3d(in_dim, out_dim, kernel_size=k, stride=s, padding=p)]

        # Normalization Layer
        if norm is True:
            layers += [nn.InstanceNorm3d(out_dim, affine=True)]

        # Non-linearity Layer
        if non_linear == 'leaky_relu':
            layers += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
        elif non_linear == 'relu':
            layers += [nn.ReLU(inplace=True)]

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_block(x)
        return out
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # Discriminator with last patch (14x14)
        # (N, 3, 128, 128) -> (N, 1, 14, 14)
        self.d_1 = nn.Sequential(nn.AvgPool3d(kernel_size=3, stride=2, padding=1, count_include_pad=False),
                                 ConvBlock(1, 32, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(32, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 # ConvBlock(64, 64, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 128, k=4, s=1, p=1, norm=False, non_linear=None),
                                 ConvBlock(128, 1, k=4, s=1, p=1, norm=False, non_linear=None))

        # Discriminator with last patch (30x30)
        # (N, 16, 128, 128) -> (N, 1, 30, 30)
        self.d_2 = nn.Sequential(ConvBlock(1, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 # ConvBlock(64, 64, k=4, s=2, p=1, norm=False, non_linear='leaky_relu'),
                                 ConvBlock(64, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 # ConvBlock(128, 128, k=4, s=2, p=1, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(128, 256, k=4, s=2, p=0, norm=True, non_linear='leaky-relu'),
                                 ConvBlock(256, 1, k=4, s=1, p=1, norm=False, non_linear=None)
                                 )

    def forward(self, x):
        out_1 = self.d_1(x)
        out_2 = self.d_2(x)
        return (out_1, out_2)
