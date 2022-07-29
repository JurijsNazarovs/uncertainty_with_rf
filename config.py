import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers',
                        type=int,
                        default=2,
                        help='number of workers for dataloaders')
    parser.add_argument('--cuda', action='store_false')
    parser.add_argument('--segm_only',
                        action='store_true',
                        default=None,
                        help="Whether to train segmentation only")
    parser.add_argument(
        '--lr_gen',
        type=float,
        default=1e-4,  #5e-5
        help='learning rate for Generator')
    parser.add_argument('--lr_disc',
                        type=float,
                        default=0.0001,
                        help='learning rate for Discriminator')
    parser.add_argument('--lr_segm',
                        type=float,
                        default=0.0001,
                        help='learning rate for Segmentation network')
    parser.add_argument('--decay_every',
                        type=int,
                        default=50,
                        help='decay interval for learning rate')
    parser.add_argument('--lr_decay',
                        type=float,
                        default=0.95,
                        help='decay rate for learning rate')
    parser.add_argument('--l2_decay',
                        type=float,
                        default=1e-5,
                        help='lambda for l2 regularization/weight decay')
    parser.add_argument('--dropout_prob',
                        type=float,
                        default=0.9,
                        help='dropout retention rate for discriminator')
    parser.add_argument('--lrelu_slope',
                        type=float,
                        default=0.2,
                        help='slope of leaky relu function')
    parser.add_argument('--opt_beta_1',
                        type=float,
                        default=0.5,
                        help='beta1 for Adam optimizer')
    parser.add_argument('--opt_beta_2',
                        type=float,
                        default=0.999,
                        help='beta2 for Adam optimizer')

    parser.add_argument(
        '--disc_type',
        type=str,
        default='conv',  #conv normal
        help=
        'Type of discriminator: conv (fully convolution) or normal (with fc layers). Default is conv.',
        choices=['conv', 'normal'])
    parser.add_argument('--crit_iter',
                        type=int,
                        default=1,
                        help='number of iterations to update discriminator')
    parser.add_argument('--gplambda',
                        type=float,
                        default=10,
                        help='GP coefficient to update discriminator')

    parser.add_argument('--flow0_n_resblocks',
                        type=int,
                        default=6,
                        help='number of residual blocks in initial flow')
    parser.add_argument('--channels',
                        type=int,
                        default=1,
                        help='number of channels in images')
    parser.add_argument('--img_size',
                        type=int,
                        default=32,
                        help='size of images')
    parser.add_argument('--latent_dim',
                        type=int,
                        default=1,
                        help='latent dimension of noise vector')
    parser.add_argument(
        '--segm_loss_weight',
        type=float,
        default=0,  #0.13,
        help='loss weight for segmentation')

    parser.add_argument(
        '--disc_loss_weight',
        type=float,
        default=1,  #0.13,
        help='loss weight for discriminator')
    parser.add_argument(
        '--gen_loss_weight',
        type=float,
        default=1,  #0.011,
        help='loss weight for generator')
    parser.add_argument(
        '--rec_loss_weight',
        type=float,
        default=1,  #0.011,
        help='loss weight for generator')

    parser.add_argument('--rec_weight_method',
                        type=str,
                        default='default',
                        help='type of loss weight for recovery loss')
    parser.add_argument('--jac_loss_weight_forw',
                        type=float,
                        default=1,
                        help='loss weight for jacobian forward')
    parser.add_argument('--jac_loss_weight_back',
                        type=float,
                        default=1,
                        help='loss weight for jacobian reverse')
    parser.add_argument('--outgrid_loss_weight_forw',
                        type=float,
                        default=1,
                        help='loss weight for outgrid forward')
    parser.add_argument('--outgrid_loss_weight_back',
                        type=float,
                        default=1,
                        help='loss weight for outgrid reverse')
    parser.add_argument(
        '--seg2_loss_weight',
        type=float,
        default=1,  #0.01
        help='loss weight for target segmentation')
    parser.add_argument(
        '--cycle_loss_weight',
        type=float,
        default=1,  #0.01
        help='loss weight for cycle')
    parser.add_argument(
        '--ident_loss_weight',
        type=float,
        default=1,  #0.01
        help='loss weight for identity')
    parser.add_argument('--freq_disc_update',
                        type=int,
                        default=1,
                        help='How ofthen to update discriminator')
    parser.add_argument('--freq_gen_update',
                        type=int,
                        default=1,
                        help='How ofthen to update generator')
    parser.add_argument('--freq_rec_update',
                        type=int,
                        default=1,
                        help='How ofthen to update reverse')
    parser.add_argument('--gan_type',
                        type=str,
                        default='lsgan',
                        choices=['gan', 'lsgan', 'wgan', 'None', 'kl'],
                        help='Type of gan')

    ###################################################
    parser.add_argument('--batch_size',
                        type=int,
                        default=100,
                        help='batch size')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=100,
                        help='number of epochs to train')

    parser.add_argument('--alpha',
                        type=int,
                        default=0.05,
                        help='Significance alpha')
    parser.add_argument('--df',
                        type=int,
                        default=5,
                        help="Degrees of freedom in RF")
    parser.add_argument('--augment',
                        action='store_true',
                        help="Whether to augment data")
    parser.add_argument('--device', type=int, default=1, help='Cuda device')
    parser.add_argument('--save',
                        type=str,
                        default='experiments/',
                        help="Path for save checkpoints")
    parser.add_argument('--disc_noise',
                        action='store_true',
                        default=None,
                        help="Whether to add noise to discriminator")
    parser.add_argument('--disc_optim',
                        default='adam',
                        help="which optimizer to use for discriminator",
                        choices=['adam', 'sgd'])
    parser.add_argument('--load',
                        action='store_true',
                        default=None,
                        help="Whether to load model")
    parser.add_argument('--best',
                        action='store_true',
                        default=None,
                        help="To load the best model")
    parser.add_argument('--batch',
                        action='store_true',
                        default=None,
                        help="To load the model from last batch")
    parser.add_argument('--load_segm_path',
                        type=str,
                        default=None,
                        help="Path for segmentaiton model")
    parser.add_argument('-r',
                        '--random-seed',
                        type=int,
                        default=1989,
                        help="Random_seed")
    parser.add_argument('--experimentID',
                        type=str,
                        default="test",
                        help='Experiment ID')
    parser.add_argument('--test_only',
                        action='store_true',
                        help='Whether only to test, no training')

    parser.add_argument('--n_epochs_start_viz',
                        type=int,
                        default=1,
                        help="When to start vizualization")
    parser.add_argument('--n_epochs_to_viz',
                        type=int,
                        default=1,
                        help="Vizualize every N epochs")
    parser.add_argument('--plots_path',
                        type=str,
                        default="./plots/",
                        help='Directory to save plots')
    parser.add_argument(
        '--data',
        type=str,
        default="synthetic",
        #choices=['synthetic', 'movmnist', 'rotmnist', 'adni'],
        help='Possible data')
    parser.add_argument('--method',
                        type=str,
                        default="between",
                        choices=['between', 'within'],
                        help='Possible method to estimate uncertainty')
    parser.add_argument('--normalize_method',
                        type=str,
                        default="scale",
                        choices=['scale', 'mean'],
                        help='Possible method to normalize data')

    parser.add_argument('--n_plots',
                        type=int,
                        default=10,
                        help="number of testings samples to plot")

    parser.add_argument('--n_plots_batch',
                        type=int,
                        default=2,
                        help="number of testings batches to plot")

    parser.add_argument('--pval_targ',
                        type=float,
                        default=0.99,
                        help="desired pval for target_model")

    parser.add_argument('--n_combine',
                        type=int,
                        default=4,
                        help="width of final combined image")

    parser.add_argument('--disc_clip_grad',
                        type=float,
                        default=0.,
                        help="clip gradient of discriminator")
    parser.add_argument('--gen_clip_grad',
                        type=float,
                        default=0.,
                        help="clip gradient of generator")
    parser.add_argument('--shuffle',
                        action='store_true',
                        help="shiffle data or not. For test is not")
    parser.add_argument('--ode_solver',
                        type=str,
                        default='euler',
                        help='ode solver')
    parser.add_argument('--ode_grad',
                        type=str,
                        default='y0',
                        choices=['y', 'y0', 'none'],
                        help='ode grad use y or y0')
    parser.add_argument('--ode_vf',
                        type=str,
                        default='y',
                        choices=[
                            'y', 'y0', 'init_img', 'warp_img', 'warp_init_img',
                            'init_img_y', 'y_y0', 'warp_img_y0',
                            'warp_img_y0_samenet'
                        ],
                        help='ode vector field use y or y0')
    parser.add_argument('--ode_step_size',
                        type=float,
                        default=0.01,
                        help='ode step in gradient')
    parser.add_argument('--ode_norm',
                        type=str,
                        default='none',
                        choices=['scale', 'clip', 'none'],
                        help='normalize ode output')
    parser.add_argument('--unet_add_input',
                        action='store_true',
                        help='add input in unet as one of skip connection')
    parser.add_argument('--last_warp',
                        action='store_true',
                        help='whether in reverse use last warp or image')
    parser.add_argument('--ode_addt',
                        action='store_true',
                        help='whether to add time aspect in modelling u(x)')

    return parser


if __name__ == '__main__':
    parser = get_arguments()
    args = parser.parse_args()
