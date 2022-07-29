import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from . import stn
import utils


def default_unet_features():
    nb_features = [
        # Unet
        #[16, 32, 64, 128],  # encoder
        #[128, 64, 32, 32, 32, 16, 16]  # decoder: 3 extra
        #
        #[16, 32, 64, 128],  # encoder
        #[256, 128, 64, 32, 32, 32, 16]  # decoder
        # Unet2
        #[64, 128, 256, 512],  # encoder
        #[512, 256, 128, 64, 64, 32, 16]  # decoder, 3 extra
        # Unet3
        [64, 128, 256, 512],  # encoder
        [1024, 512, 256, 128, 64, 32]  # decoder, 2 extras 
        # Unet 4
        #[64, 128, 256, 512, 1024],  # encoder
        #[1024, 512, 256, 128, 64]  # decoder, 0 extras
    ]
    return nb_features


class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """
    def __init__(self,
                 inshape,
                 inch=1,
                 nb_features=None,
                 nb_levels=None,
                 feat_mult=1,
                 latent_info_ch=0,
                 add_input=False):
        super().__init__()
        """
        Parameters:
        -inshape: Input shape. e.g. (192, 192, 192)
        -nb_features: Unet convolutional features. Can be specified via a
        list of lists with
        the form [[encoder feats], [decoder feats]], or as a single
        integer. If None (default),
        the unet features are defined by the default config described
        in the class documentation.
        -nb_levels: Number of levels in unet. Only used when nb_features is
        an integer. Default is None.
        -feat_mult: Per-level feature multiplier. Only used when nb_features
        is an integer. Default is 1.
        -add_input: if concatenate input as one of the las skip connection
        """

        # ensure correct dimensionality
        self.add_input = add_input
        ndims = len(inshape)
        assert ndims in [
            1, 2, 3
        ], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # Default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # Build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError(
                    'must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features *
                             feat_mult**np.arange(nb_levels)).astype(int)
            self.enc_nf = feats[:-1]
            self.dec_nf = np.flip(feats)
        elif nb_levels is not None:
            raise ValueError(
                'cannot use nb_levels if nb_features is not an integer')
        else:
            self.enc_nf, self.dec_nf = nb_features

        if self.add_input:
            assert len(self.dec_nf) >= len(
                self.enc_nf
            ) + 1, "Add_input is used, then Decoder should have features >= then encoder + 1"
        else:
            assert len(self.dec_nf) >= len(
                self.enc_nf), "Decoder should have features >= then encoder"

        # [1] Configure Encoder (down-sampling path)
        self.inch = inch
        prev_nf = inch  # start with input channlsl
        self.downarm = nn.ModuleList()
        for nf in self.enc_nf:
            self.downarm.append(ConvBlock(
                ndims, prev_nf, nf, stride=2))  #str=2 in voxelmorph paper
            prev_nf = nf

        # [1'] Configure Appending latent information (in case we have lat img or t)
        if latent_info_ch > 0:
            # 1) We need to Flatten resulted down-sampling
            # 2) Append latent information
            # 3) Apply FC net (fc_latent)
            # 4) Recover shape for Decoder (view?)
            latent_info_out_ch = 1024
            self.latent_net = nn.Sequential(
                # * 4 because of Flattening from Conv and back to Conv
                nn.Linear(prev_nf * 4 + latent_info_ch, 1024),
                nn.ReLU(),
                nn.Linear(1024, latent_info_out_ch * 4))
            prev_nf = latent_info_out_ch

        else:
            self.latent_net = None

        # [2] Configure Decoder (up-sampling path)
        enc_history = list(reversed(self.enc_nf))
        # Combination of upsampling and conv with stride 1 is equivalent
        # to transpose convolution with stride 2, but better in performance
        # https://stackoverflow.com/questions/48226783/what-is-the-difference-between-performing-upsampling-together-with-strided-trans
        # https://blog.keras.io/building-autoencoders-in-keras.html
        # https://distill.pub/2016/deconv-checkerboard/
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.uparm = nn.ModuleList()

        for i, nf in enumerate(self.dec_nf[:len(self.enc_nf)]):
            # First decoder block (upsamle path) we do not append
            # history, but perform convolutions and upsampling.
            # Convolution done with stride 1 + padding, so
            # it is the same dimension as enc_history[0]. That
            # is why upsampling of x will be of shape enc_history[1].
            # That is why appending starts with enc_history[1],
            # second last encoder.
            channels = prev_nf + enc_history[i] if i > 0 else prev_nf
            self.uparm.append(ConvBlock(ndims, channels, nf, stride=1,
                                        bn=True))
            prev_nf = nf

        # [3] Configure extra Decoder convolutions (no up-sampling)
        if self.add_input:
            prev_nf += inch  # we append input (inchxWxH) in forward
        self.extras = nn.ModuleList()
        if len(self.dec_nf) > len(self.enc_nf):
            for nf in self.dec_nf[len(self.enc_nf):-1]:
                self.extras.append(
                    ConvBlock(ndims,
                              prev_nf,
                              nf,
                              stride=1,
                              bn=True,
                              activation=True))
                prev_nf = nf
            self.extras.append(
                ConvBlock(ndims,
                          prev_nf,
                          self.dec_nf[-1],
                          stride=1,
                          bn=True,
                          activation=False))

    def forward(self, x, latent_info=None):
        # [1] Encoder - down arm
        x_enc = [x]
        for layer in self.downarm:
            x_enc.append(layer(x_enc[-1]))

        x = x_enc.pop()

        # Appending and processing Latent information
        if self.latent_net is not None:
            if latent_info is None:
                raise ValueError("Latent network is defined in UNET, "
                                 "but no latenet input is provided")
            prelatent_shape = x.shape[2:]
            x = x.reshape(x.shape[0], -1)
            x = torch.cat([x, latent_info], dim=1)
            x = self.latent_net(x)
            x = x.reshape((x.shape[0], -1) + prelatent_shape)

        # [2] Decoder - up arm
        for i, layer in enumerate(self.uparm):
            x = layer(x)
            x = self.upsample(x)

            if i != len(self.uparm) - 1 or self.add_input:
                x = torch.cat([x, x_enc.pop()], dim=1)

        # [3] Extra convs at full resolution
        for layer in self.extras:
            x = layer(x)
        return x

    def print_layout(self):
        n_levels = len(self.downarm)
        left = []
        right = []

        downch = [i.main.out_channels for i in self.downarm]
        upch = [i.main.out_channels for i in self.uparm]
        extrch = [i.main.out_channels for i in self.extras]

        # First block with input channels
        if self.add_input:
            left.append("%s" % self.inch)
            right.append("%s + %s" % (self.inch, upch[-1]))

        # Middle blocks
        print(upch)
        for i in range(n_levels - 1):
            left.append("%s" % downch[i])
            right.append("%s + %s --> %s" %
                         (downch[i], upch[-(i + 2)], upch[-(i + 1)]))

        # Bottom block
        left.append("%s -- > " % downch[-1])
        right.append("%s" % upch[0])

        # Extra convolutions after unet
        for i in extrch:
            right[0] += ' -e-> %s' % i

        # Output
        output = [''] * len(right)
        padding = ""
        for i in range(len(output) - 1, -1, -1):
            output[i] = "%s%s%s" % (left[i], padding, right[i])
            padding = ' ' + '-' * (len(output[i])) + ' '

        for i in range(len(output)):
            if i > 0:
                outside_padding = len(output[0]) - len(output[i])
                _left = outside_padding // 3
                left_padding = ''.join([' '] * _left)
                _right = outside_padding - _left
                right_padding = ''.join([' '] * _right)
                output[i] = ''.join([left_padding, output[i], right_padding])
            print(output[i])


class VxmDense(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(
        self,
        inshape,
        inch=1,
        include_latent_img=False,
        include_latent_t=False,
        ndims=None,
        nb_unet_features=None,
        nb_unet_levels=None,
        unet_feat_mult=1,
        unet_add_input=False,
    ):
        """ 
        Parameters:
        -inshape: Input shape. e.g. (192, 192, 192)
        -inch: Input channels, e.g. image: 1, image + coordinates: 3    
        -ndims: number of dimensions to run unet, e.g. 2 => conv2d, 3 => conv3d
        -nb_unet_features: Unet convolutional features.
            Can be specified via a list of lists with  
            the form [[encoder feats], [decoder feats]], or as a single integer.
            If None (default),
            the unet features are defined by the default config described in
            the unet class documentation.
            nb_unet_levels: Number of levels in unet.
            Only used when nb_features is an integer. Default is None.
            unet_feat_mult: Per-level feature multiplier.
            Only used when nb_features is an integer. Default is 1.
        """
        super().__init__()

        # ensure correct dimensionality
        if ndims is None:
            ndims = len(inshape)
        assert ndims in [
            1, 2, 3
        ], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        if include_latent_img:
            latent_img_ch = 256
            self.latent_img_net = Encoder2d(1, latent_img_ch)
        else:
            latent_img_ch = 0
            self.latent_img_net = None
        if include_latent_t:
            latent_t_ch = 128
            latent_t_net = [
                nn.Linear(1, 32),
                nn.ReLU(),
                nn.Linear(32, 64),
                nn.ReLU(),
                nn.Linear(64, latent_t_ch)
            ]
            self.latent_t_net = nn.Sequential(*latent_t_net)
        else:
            latent_t_ch = 0
            self.latent_t_net = None

        # configure core unet model
        self.unet = Unet(inshape,
                         inch=inch,
                         nb_features=nb_unet_features,
                         nb_levels=nb_unet_levels,
                         feat_mult=unet_feat_mult,
                         latent_info_ch=latent_img_ch + latent_t_ch,
                         add_input=unet_add_input)
        self.unet.print_layout()

        #configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        #self.flow = Conv(self.unet.dec_nf[-1], 2, kernel_size=3, padding=1)
        self.flow = Conv(self.unet.dec_nf[-1], 2, kernel_size=1, padding=0)

    def forward(self, source, t=None, img=None):
        '''
        Parameters:
            source: Source image tensor.
        '''
        latent_info = []
        if self.latent_img_net is not None:
            if img is None:
                raise ValueError("latent_img_net is built, but img is None")

            latent_img = self.latent_img_net(img)
            latent_info.append(latent_img)

        if self.latent_t_net is not None:
            if t is None:
                raise ValueError("latent_img_net is built, but t is None")
            latent_t = self.latent_t_net(t)
            latent_info.append(latent_t)
        if len(latent_info) > 0:
            latent_info = torch.cat(latent_info, axis=1)
        else:
            latent_info = None

        x = self.unet(source, latent_info)
        flow_field = self.flow(x)

        return flow_field


class ConvBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """
    def __init__(self,
                 ndims,
                 in_channels,
                 out_channels,
                 stride=1,
                 bn=True,
                 activation=True):
        super().__init__()

        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.main = Conv(in_channels, out_channels, 3, stride, 1)
        if bn:
            self.bn = nn.BatchNorm2d(out_channels, track_running_stats=False)
        else:
            self.bn = None
        if activation:
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = None

    def forward(self, x):
        out = self.main(x)
        if self.bn is not None:
            out = self.bn(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


# ===== 2d Encoder/Decoder - cnn+fc ===== #
class Encoder2d(nn.Module):
    def __init__(self, input_dim, output_dim, ks=3):
        super(Encoder2d, self).__init__()
        # Encodoe 2d image in 1d data for RNN

        layers = nn.Sequential(
            nn.Conv2d(input_dim, 12, ks, stride=1, padding=1),  # 32, 32
            nn.ReLU(),  #
            nn.Conv2d(12, 24, ks, stride=2, padding=1),  # 16, 16
            nn.ReLU(),
            nn.Conv2d(24, output_dim, ks, stride=2,
                      padding=1),  # output_dim, 8, 8
            #nn.Sigmoid(),
            nn.Flatten(2),  #output_dim, 64
            nn.Linear(64, 1),  #output_dim, 1
            nn.Flatten(1)  #output_dim
        )

        utils.init_network_weights(layers)
        self.layers = layers

    def forward(self, x):
        x = self.layers(x)
        return x
