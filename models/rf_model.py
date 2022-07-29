#from models.Discriminator import DiscriminatorStrong as Discriminator
#from models.Generator import Generator
#from models.stn_affine import Stnet
from models.stn import SpatialTransformer
from models.node import STODE
#from models.Classifier import Classifier

from models.Segmentation import UNet
#import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import utils


def weights_init(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


class RFModel():
    def __init__(self, args, device='cpu'):
        if args.disc_type == "conv":
            from models.Discriminator import DiscriminatorFConv as Discriminator
        else:
            from models.Discriminator import Discriminator as Discriminator

        self.discr = Discriminator(args).to(device)
        self.gen = STODE(args, device).to(device)

        self.segm = UNet(1, 1).to(device)  #segmentation network
        self.device = device

        self.discr.apply(weights_init)
        self.gen.apply(weights_init)
        self.segm.apply(weights_init)

        if args.gan_type == 'gan':
            self.gan_crit = nn.BCELoss  # no ()
        elif args.gan_type == 'lsgan':
            self.gan_crit = nn.MSELoss  #lsGan
        else:
            self.gan_crit = nn.MSELoss  #None

        self.class_crit = nn.BCEWithLogitsLoss  # BSELoss but then need sigmoid

    def get_mask(self, x, q=0.5, is_source=True, is_transform=False):
        if is_source:
            # Get target image
            init_img = x.clone()
            x = self.gen(x, backward=False)  #ODE

        mask = torch.round(torch.sigmoid(self.segm(x)) - (q - 0.5))

        if is_transform:
            # From Target to Source
            self.gen.stode_grad.mask_init = mask.clone()
            _ = self.gen(x, backward=True, init_img=init_img)

            mask = self.gen.stode_grad.mask
            self.gen.stode_grad.mask_init = None

        return mask.to(torch.int64)

    def get_n_parameters(self):
        total = 0
        for model in [self.gen, self.discr]:
            total += sum(p.numel() for p in model.parameters()
                         if p.requires_grad)

        print("Total number of parameters: %d" % total)
