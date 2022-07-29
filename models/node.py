import time
import numpy as np

import torch
import torch.nn as nn

# git clone https://github.com/rtqichen/torchdiffeq.git
from torchdiffeq import odeint as odeint  # computes gradient insider forward on each step
#from torchdiffeq import odeint_adjoint as odeint  #ignores gradient inside the function

import utils
from models.stn import SpatialTransformer
import matplotlib.pyplot as plt
import os
from models.voxelmorph_mine import VxmDense as GradNet


class STODEGradNet(nn.Module):
    '''
    Class to model derivative of ODE, which generates 
    spatial transformation solution, and applies it to self.img
    '''
    def __init__(self, args):
        super(STODEGradNet, self).__init__()

        if args.ode_vf == 'y' or args.ode_vf == 'y_y0':
            inch = 2
        elif args.ode_vf == 'warp_init_img':
            inch = 2 * args.channels
        elif args.ode_vf == 'init_img_y':
            #Combine initial image and coordinate system (2 channels)
            inch = args.channels + 2
        else:
            inch = 1

        net = GradNet((args.img_size, args.img_size),
                      inch=inch,
                      include_latent_img=args.ode_vf == "y",
                      include_latent_t=args.ode_addt,
                      unet_add_input=args.unet_add_input)
        if args.ode_vf == 'y_y0' or args.ode_vf == 'warp_img_y0':
            net2 = GradNet((args.img_size, args.img_size),
                           inch=1,
                           include_latent_img=False,
                           include_latent_t=False,
                           unet_add_input=args.unet_add_input)
            self.net2 = net2

        self.net = net
        self.backward = False
        # Following optimizer is to optimize mid training: Failed approach
        self.optimizer = torch.optim.Adam(net.parameters(),
                                          lr=args.lr_gen,
                                          weight_decay=args.l2_decay,
                                          betas=(args.opt_beta_1,
                                                 args.opt_beta_2))

        self.img = None
        self.imgtowarp = None
        self.warpimg = None
        self.jacob = torch.zeros(1, requires_grad=True)
        self.outgrid_loss = torch.zeros(1, requires_grad=True)

        self.stn = SpatialTransformer(size=[args.img_size] * 2)
        self.mask_init = None
        self.mask = None

        self.nfe = 0

        self.ode_vf = args.ode_vf
        self.ode_grad = args.ode_grad
        self.ode_norm = args.ode_norm
        self.plot_seq = args.test_only
        self.model_name = ""  #args.experimentID
        self.plots_path = args.plots_path

        # Grid for jacobian
        self.img_size = args.img_size
        size = args.img_size  #32 64
        shape = [size, size]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0).type(torch.FloatTensor)

        self.grid = grid

    def forward(self, t, y):
        # Model: Generate vector field approach
        # y is vector field at time t.
        t_local = t * torch.ones((y.shape[0], 1)).to(y)
        self.warpimg = self.stn(self.imgtowarp, y)

        # Vector field based on selected method
        if self.ode_vf == 'y':
            vf = self.net(y, t_local, img=self.img)
        elif self.ode_vf == 'init_img':
            vf = self.net(self.img, t_local)
        elif self.ode_vf == 'warp_img':
            vf = self.net(self.warpimg, t_local)
        elif self.ode_vf == 'init_img_y':
            # Combine coordinates and initial image
            comb = torch.cat((self.img, y), dim=1)
            vf = self.net(comb, t_local)
        elif self.ode_vf == 'warp_init_img':
            # Combine  initial image and warped image
            comb = torch.cat((self.img, self.warpimg), dim=1)
            vf = self.net(comb, t_local)
        elif self.ode_vf == 'y_y0':
            vf = self.net(y, t_local, img=self.img) +\
                self.net2(self.img, t_local)
        elif self.ode_vf == 'warp_img_y0':
            vf = self.net(self.warpimg, t_local) +\
                self.net2(self.img, t_local)
        elif self.ode_vf == 'warp_img_y0_samenet':
            vf = self.net(self.warpimg, t_local) +\
                self.net(self.img, t_local)
        else:
            raise ValueError("Unknown type of ode_vf: %s" % self.ode_vf)

        grad = vf  # this is change in vector field, that is delta vf
        self.jacob = self.jacob + utils.jacdet_loss(
            vf, self.grid, backward=self.backward)
        self.outgrid_loss = self.outgrid_loss + utils.outgrid_loss(
            vf, self.grid, backward=self.backward, size=self.img_size)

        if self.ode_norm == 'scale':
            self.warpimg = utils.normalize(self.warpimg, method='scale')
        elif self.ode_norm == 'clip':
            self.warpimg = torch.clip(self.warpimg, 0, 1)

        plot_ind = 0
        if self.mask_init is not None:
            # Warping mask from target to source
            if self.mask is None:
                self.mask = self.mask_init.clone()
            self.mask = self.stn(self.mask_init, y)
            self.mask = torch.ceil(self.mask)

            # Below is plotting of warped resutls per step: mask and image
            if self.nfe < 300 and self.nfe >= 200:
                os.makedirs("%s/mask_seq" % self.plots_path, exist_ok=True)
                plt.imshow(self.mask[plot_ind, 0].cpu().numpy(),
                           cmap='bwr',
                           interpolation=None)
                plt.clim(-1, 1)
                plt.title("Step: %04d" % self.nfe)
                plt.savefig("%s/mask_seq/yhat_%04d.png" %
                            (self.plots_path, self.nfe))
                plt.close()

            # Plot masked image
            if self.nfe < 300 and self.nfe > 200:
                img_masked = np.ma.masked_where(self.mask.cpu().numpy() < 1,
                                                self.warpimg.cpu().numpy())
                os.makedirs("%s/img_mask_seq" % self.plots_path, exist_ok=True)
                plt.imshow(
                    self.warpimg[plot_ind, 0].cpu().numpy(),
                    #cmap='Greys',
                    cmap='jet',
                    interpolation='bilinear')
                plt.clim(0, 1)

                if True:  #empty_plot:
                    plt.axis('off')
                else:
                    plt.colorbar()

                plt.imshow(img_masked[plot_ind, 0],
                           cmap='binary',
                           interpolation='none')
                plt.clim(0, 1)
                if self.nfe == 201:
                    plt.title("Mask allocation on GRF")
                elif self.nfe == 299:
                    plt.title("Mask on Source Image")
                else:
                    plt.title("Step: %04d" % (self.nfe - 101))

                if True:  #empty_plot:
                    plt.axis('off')
                else:
                    plt.colorbar()

                #plt.colorbar()
                plt.savefig("%s/img_mask_seq/yhat_%04d.png" %
                            (self.plots_path, self.nfe))
                plt.close()

        self.nfe += 1
        if self.plot_seq and self.nfe < 200 and not y.requires_grad:
            plots_path = "%s/vf_seq/%s/" % (self.plots_path, self.model_name)
            os.makedirs(plots_path, exist_ok=True)

            if self.warpimg.shape[1] > 1:
                # plots rgb image
                plt.imshow(self.warpimg[plot_ind].cpu().numpy().transpose(
                    (1, 2, 0)),
                           interpolation='bilinear',
                           cmap='jet')
            else:
                plt.imshow(self.warpimg[plot_ind, 0].cpu().numpy(),
                           interpolation='bilinear',
                           cmap='jet')
            plt.title("Step: %04d" % self.nfe)

            if True:  #empty_plot:
                plt.axis('off')
            else:
                plt.colorbar()

            plt.savefig("%s/yhat_%04d.png" % (plots_path, self.nfe))
            plt.close()

        return grad


class DiffeqSolver(nn.Module):
    def __init__(self,
                 ode_grad_net,
                 method='euler',
                 odeint_rtol=1e-4,
                 odeint_atol=1e-5,
                 ode_step_size=0.01):
        super(DiffeqSolver, self).__init__()

        self.ode_method = method
        self.ode_grad_net = ode_grad_net
        self.odeint_rtol = odeint_rtol
        self.odeint_atol = odeint_atol
        self.ode_step_size = ode_step_size

    def forward(self, first_point, time_steps_to_predict):
        def _grid_constructor(func, y0, t):
            t_infer = torch.tensor([t[0], t[-1]]).to(y0)

            return t_infer

        pred_y = odeint(
            self.ode_grad_net,
            first_point,
            time_steps_to_predict,
            rtol=self.odeint_rtol,
            atol=self.odeint_atol,
            method=self.ode_method,
            options={
                'step_size': self.ode_step_size,  #0.01,
                #'grid_constructor': _grid_constructor
            })

        return pred_y[-1]  #return last point only - last flow

    @property
    def nfe(self):
        # Number of forward estimates
        return self.ode_grad_net.nfe

    @nfe.setter
    def nfe(self, value):
        self.ode_grad_net.nfe = value


SOLVERS = [
    "dopri5", "bdf", "rk4", "midpoint", 'adams', 'explicit_adams',
    'fixed_adams'
]

#rk4 is fixed timegrid method to avoid dynamic allocation of memory
#dopri5 is adaptive step method


class STODE(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.stode_grad = STODEGradNet(args).to(device)
        self.de_solver = DiffeqSolver(self.stode_grad,
                                      method=args.ode_solver,
                                      odeint_rtol=1e-3,
                                      odeint_atol=1e-4,
                                      ode_step_size=args.ode_step_size)

        self.last_warp = args.last_warp
        self.model_name = ""
        self.plots_path = args.plots_path

    def forward(self,
                x,
                backward=False,
                normalize=True,
                y=None,
                init_img=None):
        return self.forward_warpode(x,
                                    backward=backward,
                                    normalize=normalize,
                                    y=y,
                                    init_img=init_img)

    def forward_warpode(self,
                        x,
                        backward=False,
                        normalize=True,
                        y=None,
                        init_img=None):
        if backward:
            t = torch.tensor([1, 0]).to(x)
        else:
            t = torch.tensor([0, 1]).to(x)
        self.stode_grad.backward = backward

        # Set initial flow as Identity transformation, i.e. zeros
        flow0_shape = list(x.shape)  #batch, channels, height, width
        #flow0_shape[1] *= 2  #for spatial transformation we need 2 coordinates
        flow0_shape[1] = 2  #for spatial transformation we need 2 coordinates
        flow0 = (torch.zeros(flow0_shape)).to(x)

        # Learn flow and warp image
        if backward:
            self.stode_grad.img = init_img.clone()
            # Note: that backward should run only after forward is ran,
            # otherwise  self.stode_grad.warpimg is None
            if not self.last_warp:
                self.stode_grad.warpimg = x.clone()  #copy image
            else:
                if self.stode_grad.warpimg is None:
                    raise ValueError()
        else:
            self.stode_grad.img = x.clone()  #copy image
            self.stode_grad.warpimg = x.clone()  #copy image
        self.stode_grad.imgtowarp = x.clone()
        self.stode_grad.jacob = torch.zeros(1, requires_grad=True).to(x)
        self.stode_grad.outgrid_loss = torch.zeros(1, requires_grad=True).to(x)
        self.stode_grad.grid = self.stode_grad.grid.to(x)

        flow = self.de_solver(flow0, t)
        y_hat, y_hat_grid = self.stode_grad.stn(x, flow, return_grid=True)

        # We need to scale jacob and outgrid_loss, since it is collected through
        # all time steps. We use 100 as a scale, because it  is #steps in ode.
        # But even if this si not, it works fine.
        self.stode_grad.jacob = self.stode_grad.jacob.sum() / 100
        self.stode_grad.outgrid_loss = self.stode_grad.outgrid_loss.sum() / 100

        if self.stode_grad.ode_norm == 'scale':
            y_hat = utils.normalize(y_hat, method='scale')
        elif self.stode_grad.ode_norm == 'clip':
            y_hat = torch.clip(y_hat, 0, 1)

        #print('y_min, y_max:', y_hat.min(), y_hat.max())
        #print('flow_min, flow_max:', flow.min(), flow.max())

        if not backward and normalize and not flow.requires_grad:
            # Reverse warping - attempt to recover x0
            # and Plotting of reverse images
            self.stode_grad.imgtowarp = y_hat.clone()
            flow_reverse = self.de_solver(flow0, torch.tensor([1, 0]).to(x))
            x_reverse, reverse_grid = self.stode_grad.stn(y_hat,
                                                          flow_reverse,
                                                          grid=y_hat_grid,
                                                          return_grid=True)

            # Plotting reverse images
            ind = 0
            step = 1
            # Plot final flow
            flow_ = flow[ind, :, ::step, ::step].detach().cpu().numpy()

            # Negative flow is used because we take values of new grid and put
            # them in the current grid, in pixels, where new grid is defined.
            # That is, new_grid[20,20]=(1,1), means in pixel (20,20) we put
            # value of pixel (1,1)
            u = -flow_[1]
            v = -flow_[0]

            #axis = np.arange(0, flow.shape[2], step)
            #x_, y_ = np.meshgrid(axis, axis)
            y_ = y_hat_grid[ind, 0].cpu().data.numpy()
            x_ = y_hat_grid[ind, 1].cpu().data.numpy()

            plots_path = "%s/vf/%s/" % (self.plots_path, self.model_name)
            os.makedirs(plots_path, exist_ok=True)
            fig, ax = plt.subplots()
            plt.quiver(x_, y_, u, v, angles='xy', scale_units='xy', scale=1)
            plt.savefig("%s/vf.png" % plots_path)
            plt.close()

            fig, ax = plt.subplots()
            plt.imshow(
                np.minimum(
                    0 * y_hat[ind, 0].detach().cpu().numpy() +
                    1 * x[ind, 0].cpu().numpy(),
                    1,
                ))
            plt.quiver(x_, y_, u, v, angles='xy', scale_units='xy', scale=1)
            plt.savefig("%s/vf_overlap.png" % plots_path)
            plt.close()

            # Reverse flow
            flow_ = flow_reverse[ind, :, ::step, ::step].detach().cpu().numpy()
            y_ = reverse_grid[ind, 0].cpu().data.numpy()
            x_ = reverse_grid[ind, 1].cpu().data.numpy()
            u = -flow_[1]
            v = -flow_[0]
            plt.quiver(x_,
                       y_,
                       u,
                       v,
                       angles='xy',
                       scale_units='xy',
                       scale=1,
                       color='red')
            plt.savefig("%s/vf_reverse.png" % plots_path)
            plt.close()

            if y is not None:
                plt.imshow(y[ind, 0].cpu().numpy() -
                           y_hat[ind, 0].detach().cpu().numpy(),
                           interpolation='bilinear',
                           cmap='jet')
                plt.savefig("%s/y_yhat.png" % plots_path)
                plt.close()

                plt.imshow(y[ind, 0].cpu().numpy(),
                           interpolation='bilinear',
                           cmap='jet')
                plt.colorbar()
                plt.savefig("%s/y.png" % plots_path)
                plt.close()

            plt.imshow(y_hat[ind, 0].detach().cpu().numpy(),
                       interpolation='bilinear',
                       cmap='jet')
            plt.clim(0, 1)
            #plt.clim(-3, 3)
            plt.savefig("%s/y_hat.png" % plots_path)
            plt.close()

            plt.imshow(x[ind, 0].cpu().numpy() -
                       y_hat[ind, 0].detach().cpu().numpy(),
                       interpolation='bilinear',
                       cmap='jet')
            plt.savefig("%s/x_yhat.png" % plots_path)
            plt.close()

            plt.imshow(x[ind, 0].cpu().numpy(),
                       interpolation='bilinear',
                       cmap='jet')
            plt.clim(0, 1)
            #plt.clim(-3, 3)
            plt.colorbar()
            plt.savefig("%s/x.png" % plots_path)
            plt.close()

            plt.imshow(x_reverse[ind, 0].cpu().numpy(),
                       interpolation='bilinear',
                       cmap='jet')
            plt.clim(0, 1)
            plt.colorbar()
            plt.savefig("%s/x_reverse.png" % plots_path)
            plt.close()

            plt.imshow(x[ind, 0].cpu().numpy() -
                       x_reverse[ind, 0].cpu().numpy(),
                       interpolation='bilinear',
                       cmap='jet')
            plt.clim(0, 1)
            plt.colorbar()
            plt.savefig("%s/x_x_reverse.png" % plots_path)
            plt.close()

        return y_hat
