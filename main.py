import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import itertools

import os
import sys
import pickle
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import time

from config import get_arguments
from data_config import get_datapath
import utils
from models.rf_model import RFModel
import rf_collection as rfc

from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, path='output.log'):
        self.terminal = sys.stdout
        self.log = open(path, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass


def train(args,
          source_loader,
          target_loader,
          model,
          optims,
          n_batches=1,
          epoch=0,
          ckpt_path='./ckpt',
          is_mask=False,
          best_loss=None,
          tb_writer=None,
          train_gen=True,
          train_disc=True,
          train_reverse=True):
    alpha = args.alpha
    device = model.device
    #-------------------------------------------------------------------
    # Define models and optimizers
    #-------------------------------------------------------------------

    generator, discriminator, classifier = model.gen, model.discr, model.segm

    if args.segm_only:
        classifier.train()
        generator.train(False)
        discriminator.train(False)
    else:
        classifier.train(False)
        generator.train()
        discriminator.train()
    optim_G, optim_D, optim_S = optims

    #-------------------------------------------------------------------
    # Define criteria loss for model
    #-------------------------------------------------------------------
    gan_crit = model.gan_crit()
    #discriminator loss
    class_weights = torch.FloatTensor([(1 - alpha) / alpha]).to(device)
    class_crit_ = model.class_crit(pos_weight=class_weights)
    class_crit = lambda x, y: class_crit_(x.squeeze(),
                                          y.squeeze().to(torch.float32))

    #-------------------------------------------------------------------
    running_segm_loss = 0.0
    running_gen_loss = 0.0
    running_rec_loss = 0.0
    running_disc_loss = 0.0
    running_disc_acc = 0.0
    running_disc_acc_truth = 0.0
    running_disc_acc_fake = 0.0
    running_jacdet_forw = 0.0
    running_jacdet_back = 0.0
    running_outgrid_forw = 0.0
    running_outgrid_back = 0.0

    for i in range(n_batches):
        #start_time_batch = time.time()
        args.rec_loss_weight = utils.get_beta(args.rec_weight_method,
                                              n_batches,
                                              i,
                                              reverse=False,
                                              weight=args.rec_loss_weight)
        print("rec_loss_weight: ", args.rec_loss_weight)

        if not args.segm_only:
            if is_mask:
                imgs_s, _ = source_loader.__next__()
            else:
                imgs_s = source_loader.__next__()

            if args.augment:
                imgs_s = utils.aug_trans(imgs_s)
            imgs_s = imgs_s.to(model.device)

        imgs_t, mask_t = target_loader.__next__()
        if args.augment and args.segm_only:
            imgs_t, mask_t = utils.aug_trans(imgs_t, mask_t)
        imgs_t = imgs_t.to(device)
        mask_t = mask_t.to(device)

        if not args.segm_only:
            N = min(imgs_t.shape[0], imgs_s.shape[0])  #batch_size
            imgs_s, imgs_t, mask_t = imgs_s[:N], imgs_t[:N], mask_t[:N]

        # Classifier (segmentation) on target images (Z)
        if args.segm_only:
            pred_t = classifier(imgs_t)
            segm_loss = class_crit(pred_t, mask_t)
            running_segm_loss = utils.get_running(running_segm_loss,
                                                  segm_loss.item(), i)
            optim_S.zero_grad()
            segm_loss.backward()
            optim_S.step()

            print("[ Batch %04d/%04d] [ Segm Loss: %05.4f ]" %
                  (i + 1, n_batches, running_segm_loss),
                  end='\n',
                  flush=True)
            # Save every batch
            utils.save_model(args, model, ckpt_path + '_batch', epoch,
                             best_loss)
            continue

        # --------------
        # Train Generator: F -> Z
        # --------------
        truth_label = torch.ones((N, 1)).to(device)
        false_label = torch.zeros((N, 1)).to(device)

        imgs_fake_t = generator(imgs_s, y=imgs_t)
        if args.gan_type == 'wgan':
            # https://arxiv.org/pdf/1704.00028.pdf page 4
            gen1_loss = -discriminator(imgs_fake_t).mean()
        elif args.gan_type in ['gan', 'lsgan']:
            gen1_loss = gan_crit(discriminator(imgs_fake_t), truth_label)
        else:
            # MSELoss/L1Loss
            gen1_loss = nn.MSELoss(reduction='sum')(imgs_fake_t, imgs_t)

        gen_loss = gen1_loss / N * args.gen_loss_weight
        running_gen_loss = utils.get_running(running_gen_loss, gen_loss.item(),
                                             i)

        # Add jacobian determinant constrain
        if args.jac_loss_weight_forw != 0:
            gen_loss += generator.stode_grad.jacob / N * args.jac_loss_weight_forw
        running_jacdet_forw = utils.get_running(
            running_jacdet_forw,
            generator.stode_grad.jacob.item() / N, i)
        print('jacob_loss forward:', generator.stode_grad.jacob / N)

        # Add outside of the grid constrain (to keep vf inside the grid)
        if args.outgrid_loss_weight_forw != 0:
            gen_loss += generator.stode_grad.outgrid_loss / N * args.outgrid_loss_weight_forw
        running_outgrid_forw = utils.get_running(
            running_outgrid_forw,
            generator.stode_grad.outgrid_loss.item() / N, i)
        print('outgrid_loss forward:', generator.stode_grad.outgrid_loss / N)

        if i == 0:
            print("\nn steps: %d" % model.gen.de_solver.nfe)

        model.gen.de_solver.nfe = 0  #reset counter of ode steps
        if train_gen:
            # -----------------------
            # Reverse Generator Start
            def get_rec_loss():
                imgs_fake_s = generator(
                    imgs_fake_t,  #.detach(),
                    y=None,
                    backward=True,
                    init_img=imgs_s)
                rec_loss_ = nn.MSELoss(reduction='sum')(imgs_fake_s, imgs_s)
                #recov_loss_ = nn.L1Loss(reduction='sum')(imgs_fake_s, imgs_s)
                rec_loss = rec_loss_ / N * args.rec_loss_weight
                # Below cannot use the same name, because it is local variable
                # so I add _ at the end: running_rec_loss_
                running_rec_loss_ = utils.get_running(running_rec_loss,
                                                      rec_loss_.item() / N, i)
                return (rec_loss, running_rec_loss_)

            if args.rec_loss_weight != 0 and train_reverse:
                recov_loss, running_rec_loss = get_rec_loss()

                # Add jacobian determinant constrain
                if args.jac_loss_weight_back != 0:
                    recov_loss += generator.stode_grad.jacob / N * args.jac_loss_weight_back
                running_jacdet_back = utils.get_running(
                    running_jacdet_back,
                    generator.stode_grad.jacob.item() / N, i)
                print('jacob_loss backward:', generator.stode_grad.jacob / N)

                # Add outside of the grid constrain (to keep vf inside the grid)
                if args.outgrid_loss_weight_back != 0:
                    recov_loss += generator.stode_grad.outgrid_loss / N * args.outgrid_loss_weight_back
                running_outgrid_back = utils.get_running(
                    running_outgrid_back,
                    generator.stode_grad.outgrid_loss.item() / N, i)
                print('outgrid_loss backward:',
                      generator.stode_grad.outgrid_loss / N)

                gen_loss += recov_loss
            else:
                # Run without gradient to keep results in summary
                with torch.no_grad():
                    recov_loss, running_rec_loss = get_rec_loss()

            # Reverse Generator End
            # -----------------------

            utils.check_grads(generator, 'Generator')
            optim_G.zero_grad()
            gen_loss.backward()

            if args.gen_clip_grad != 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(),
                                               args.gen_clip_grad)
            optim_G.step()

        # --------------
        # Train Discriminator,
        # by fixing \theta_G, \theta_C, and optimize for \theta_D
        # --------------
        for crit_iter in range(args.crit_iter):

            truth_label = torch.ones((N, 1)).to(device)
            false_label = torch.zeros((N, 1)).to(device)
            ## ---------- Tricks to train GAN ----------
            # Add noise labels eacho N iterations (by swapping lables)
            if i % 10 == 0:
                noise_p = 0.15
                ind_add_noise = np.random.choice(
                    len(truth_label), int(noise_p * len(truth_label)))
                truth_label[ind_add_noise] = false_label[ind_add_noise]

                ind_add_noise = np.random.choice(
                    len(truth_label), int(noise_p * len(truth_label)))
                false_label[ind_add_noise] = truth_label[ind_add_noise]
            ## ---------- Tricks to train GAN END ----------

            if args.gan_type == 'wgan':
                gradient_penalty = utils.calc_gradient_penalty(
                    discriminator,
                    imgs_t,
                    imgs_fake_t.detach(),
                    LAMBDA=args.gplambda)

                ld = -discriminator(imgs_t).mean() +\
                    discriminator(imgs_fake_t.detach()).mean()+ gradient_penalty
            elif args.gan_type in ['gan', 'lsgan']:
                ld = gan_crit(discriminator(imgs_t), truth_label) +\
                    gan_crit(discriminator(imgs_fake_t.detach()), false_label)
            else:
                ld = torch.zeros(1).to(imgs_t)

            disc_loss = ld * args.disc_loss_weight
            if torch.isnan(disc_loss):
                import pdb
                pdb.set_trace()
                print("DEBUG! Disc loss is nan")

            if train_disc and args.gan_type != 'None':
                # We skip this part in case it is L2 loss and not Gan
                utils.check_grads(discriminator, 'Discriminator')
                optim_D.zero_grad()
                disc_loss.backward()
                if True:  #disc_loss.item() / N > 0.0003:  #0.009
                    if args.disc_clip_grad != 0:
                        torch.nn.utils.clip_grad_norm_(
                            discriminator.parameters(), args.disc_clip_grad)
                    optim_D.step()
                else:
                    print('No discriminator update')
                    optim_D.zero_grad()
                running_disc_loss = utils.get_running(running_disc_loss,
                                                      disc_loss.item() / N, i)

        #end_time_batch = time.time()
        #print("Batch_time: %f" % (end_time_batch - start_time_batch))

        # --------------
        # Summaries
        # --------------
        if args.gan_type == 'wgan':
            func = lambda x: torch.round(torch.clip(x, 0, 1))
        elif args.gan_type == 'lsgan':
            func = lambda x: torch.round(torch.clip(x, 0, 1))
        else:
            func = lambda x: torch.round(x)

        disc_acc_truth = (
            (func(discriminator(imgs_t.detach()))
             == torch.round(truth_label)).sum()) / len(truth_label)

        disc_acc_fake = (
            (func(discriminator(imgs_fake_t.detach()))
             == torch.round(false_label)).sum()) / len(false_label)

        disc_acc = ((func(discriminator(imgs_t.detach())) == torch.round(truth_label)).sum() +\
            (func(discriminator(imgs_fake_t.detach())) ==\
             torch.round(false_label)).sum())/(len(truth_label) + len(false_label))
        running_disc_acc = utils.get_running(running_disc_acc, disc_acc, i)
        running_disc_acc_truth = utils.get_running(running_disc_acc_truth,
                                                   disc_acc_truth, i)
        running_disc_acc_fake = utils.get_running(running_disc_acc_fake,
                                                  disc_acc_fake, i)

        print(
            "[ Batch %04d/%04d] [ Gen Loss: %05.8f ] [ Disc Loss:  %05.8f ] [ Disc acc:  %05.4f ] [ Running Disc acc:  %05.4f ] [ Rec Loss: %05.4f]"
            % (i + 1, n_batches, running_gen_loss, running_disc_loss, disc_acc,
               running_disc_acc, running_rec_loss),
            end='\n',
            flush=True)
        # Save every batch
        utils.save_model(args, model, ckpt_path + '_batch', epoch, best_loss)

    # Save every epochs
    print()
    utils.save_model(args, model, ckpt_path, epoch, best_loss)

    if best_loss is None:
        best_loss = np.infty

    if not args.segm_only:
        # write loss to tb
        if tb_writer is not None:
            tb_writer.add_scalar('gen_loss', running_gen_loss, epoch)
            tb_writer.add_scalar('discr_loss', running_disc_loss, epoch)
            tb_writer.add_scalar('discr_acc', running_disc_acc, epoch)
            tb_writer.add_scalar('discr_acc_truth', running_disc_acc_truth,
                                 epoch)
            tb_writer.add_scalar('discr_acc_fake', running_disc_acc_fake,
                                 epoch)
            tb_writer.add_scalar('rec_loss', running_rec_loss, epoch)
            tb_writer.add_scalar('jacdet_forw', running_jacdet_forw, epoch)
            tb_writer.add_scalar('jacdet_back', running_jacdet_back, epoch)
            tb_writer.add_scalar('outgrid_forw', running_outgrid_forw, epoch)
            tb_writer.add_scalar('outgrid_back', running_outgrid_back, epoch)

    return best_loss


def test(args,
         dataloader,
         model,
         compute_summary=True,
         make_plots=True,
         best_loss=None,
         n_batches=1,
         max_n_plots=1,
         epoch=0,
         plots_path="../plots/",
         synthetic_path=None,
         ckpt_path='./ckpt',
         base_loader=None,
         is_mask=True,
         is_source=True,
         condition=True,
         testloader=None):
    print("Start testing")
    torch.cuda.empty_cache()

    generator = model.gen
    classifier = model.segm
    generator.eval()
    classifier.eval()

    summary = {}
    test_loss = 0
    test_loss_ = 0

    for batch_iter in range(n_batches):
        if is_mask:
            imgs, mask = dataloader.__next__()
            imgs = imgs.to(model.device)
            mask = mask.to(model.device)
        else:
            imgs = dataloader.__next__().to(model.device)
            mask = None

        if is_source:
            mask_hat = model.get_mask(imgs, is_source=True, is_transform=True)
        else:
            mask_hat = model.get_mask(imgs,
                                      is_source=False,
                                      is_transform=False)

        mask_cochran = None
        mask_q = None

        if synthetic_path is not None and batch_iter < args.n_plots_batch and is_source:
            if testloader is not None:
                test_imgs = testloader.__next__()
                if isinstance(test_imgs, list):
                    test_imgs = test_imgs[0]  #[1] is mask
            else:
                test_imgs = None

            imgs_fake = generator(imgs, backward=not is_source, y=test_imgs)
            mask_fake = model.get_mask(imgs_fake,
                                       is_source=not is_source,
                                       is_transform=not is_source)
        else:
            imgs_fake = None
            mask_fake = None

        if compute_summary:
            bs = mask_hat.shape[0]
            if mask is not None:
                # Compute summaries given known mask
                summary = utils.get_binary_summary(mask.reshape(bs, -1),
                                                   mask_hat.reshape(bs, -1),
                                                   summary, n_batches, "hat")

                if batch_iter == n_batches - 1:
                    #precision  recall(+)  f1-score
                    #test_loss = -summary['hat']['power']  #consistent with loss
                    if args.segm_only:
                        test_loss = -summary['hat']['iou']
                    elif args.method == 'within':
                        if is_source:
                            test_loss = -summary['hat']['iou']
                        else:
                            test_loss = -summary['hat']['pval_model']
                    else:
                        test_loss = np.abs(args.alpha -
                                           summary['hat']['pval_model'])
                    # test_loss = np.abs(summary['hat']['pval_true'] -
                    #                    summary['hat']['pval_model'])
                    type1error = summary['hat']['alpha']
            else:
                # Compute pvalue for model as percentage of rejected RF
                if args.n_combine > 1:
                    pval_model = (mask_hat.reshape(-1).detach().cpu().numpy()
                                  == 1).any()

                else:
                    pval_model = np.mean((mask_hat.reshape(
                        bs, -1).detach().cpu().numpy() == 1).any(axis=1))

                if 'hat' not in summary.keys():
                    summary['hat'] = {}
                    summary['hat']['pval_model'] = pval_model / n_batches
                else:
                    summary['hat']['pval_model'] += pval_model / n_batches

                if batch_iter == n_batches - 1:
                    if args.method == 'within':
                        test_loss = -summary['hat']['pval_model']
                    else:
                        test_loss = np.abs(args.alpha -
                                           summary['hat']['pval_model'])

                    print(test_loss)
                    type1error = None

            # Quantile mask (1-alpha)
            # if combine images then need to compute on the whole batch
            mask_q = rfc.get_significant(
                imgs,
                test="quantile",
                alpha=0.01,  #args.alpha,
                is_batch=args.n_combine < 2)

            if False:  #mask is not None:
                summary = utils.get_binary_summary(mask.reshape(bs, -1),
                                                   mask_q.reshape(bs, -1),
                                                   summary, n_batches, "q")

        if make_plots and batch_iter < args.n_plots_batch:
            if base_loader is not None:
                base_imgs = base_loader.__next__()
            print("Making plots")
            print("-----------------")
            if args.n_combine > 1:
                n_comb = args.n_combine
                imgs = utils.combine_batch(imgs, n_comb)

                if is_mask:
                    mask = utils.combine_batch(mask, n_comb)
                mask_hat = utils.combine_batch(mask_hat, n_comb)
                mask_q = utils.combine_batch(mask_q, n_comb)
                if base_loader is not None:
                    #import pdb; pdb.set_trace()
                    base_imgs = utils.combine_batch(base_imgs, n_comb)

                if imgs_fake is not None:
                    imgs_fake = utils.combine_batch(imgs_fake, n_comb)
                    mask_fake = utils.combine_batch(mask_fake, n_comb)

            for i in range(min(len(imgs), max_n_plots)):
                imgs_ = imgs[i]
                mask_hat_ = mask_hat[i]
                mask_ = mask[i] if mask is not None else None
                imgs_fake_ = imgs_fake[i] if imgs_fake is not None else None
                mask_cochran_ = mask_cochran[i] if mask_cochran is not None\
                    else None
                mask_q_ = mask_q[i] if mask_q is not None else None

                if base_loader is not None:
                    base_imgs_ = base_imgs[i]
                else:
                    base_imgs_ = None

                utils.make_plots(
                    imgs_,
                    mask_hat_,
                    mask_,
                    #imgs_fake_,
                    plots_path=plots_path + "%02d_%02d" % (batch_iter, i),
                    mask_cochran=mask_cochran_,
                    mask_quantile=mask_q_,
                    base_img=base_imgs_)

                if mask_fake is not None:
                    utils.make_plots(imgs_fake[i],
                                     mask_fake[i],
                                     plots_path=plots_path + "%02d_%02d_fake" %
                                     (batch_iter, i))

    # ------------------------------------------------------------------------
    # Print collected through testing summary
    # ------------------------------------------------------------------------
    print("-----------------")
    for key_, item_ in summary.items():
        print("%s:" % key_)
        for key, item in summary[key_].items():
            if key == 'conf_mat':
                continue
                print("%s:" % key)
                print(item)
            else:
                print("%s:" % key, end=' ')
                print("%.4f" % item, end=', ')
        print("\n-----------------")

    # ------------------------------------------------------------------------
    # Save best loss
    # ------------------------------------------------------------------------
    if best_loss is not None and compute_summary:
        if test_loss < best_loss and condition:
            best_loss = test_loss
            utils.save_model(args, model, ckpt_path + '_best', epoch,
                             best_loss)
            print("New best loss: %f" % best_loss)
        else:
            print("Best loss is still the same: %f" % best_loss)

        return best_loss

    return test_loss


def main():
    parser = get_arguments()
    args = parser.parse_args()
    data_path = get_datapath(args)
    if "synthetic" in args.data:
        args.plots_path += "%s/df%d/%s/" % (args.data, args.df, args.method)
    else:
        args.plots_path += "%s/%s/" % (args.data, args.method)

    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    os.makedirs(args.save, exist_ok=True)

    experimentID = args.experimentID
    if experimentID is None:
        experimentID = int(SystemRandom().random() * 100000)
    ckpt_path = os.path.join(args.save,
                             "experiment_" + str(experimentID) + '.ckpt')
    #os.makedirs("results/", exist_ok=True)

    #-------------------------------------------------------------------
    # Model and optimizers
    #-------------------------------------------------------------------

    if args.test_only:
        args.ode_step_size = 0.01
        args.ode_solver = 'euler'
        args.shuffle = False
        if args.n_combine > 1:
            args.batch_size = args.n_combine

    if torch.cuda.is_available():
        device = torch.device('cuda')
        if args.device:
            device = "%s:%d" % (device, args.device)
    else:
        device = torch.device('cpu')
    print("Device:", device)

    model = RFModel(args, device)
    model.get_n_parameters()

    optim_G = optim.Adam(model.gen.parameters(),
                         lr=args.lr_gen,
                         weight_decay=args.l2_decay,
                         betas=(args.opt_beta_1, args.opt_beta_2))
    if args.disc_optim == 'adam':
        print("***************")
        print("Adam optimizer for discriminator")
        print("***************")
        optim_D = optim.Adam(model.discr.parameters(),
                             lr=args.lr_disc,
                             weight_decay=args.l2_decay,
                             betas=(args.opt_beta_1, args.opt_beta_2))
    else:
        print("***************")
        print("SGD optimizer for discriminator")
        print("***************")
        optim_D = optim.SGD(model.discr.parameters(), lr=args.lr_disc)
    optim_S = optim.Adam(model.segm.parameters(),
                         lr=args.lr_segm,
                         weight_decay=args.l2_decay,
                         betas=(args.opt_beta_1, args.opt_beta_2))

    optims = [optim_G, optim_D, optim_S]
    scheduler_G = optim.lr_scheduler.StepLR(optim_G,
                                            step_size=args.decay_every,
                                            gamma=args.lr_decay)
    scheduler_D = optim.lr_scheduler.StepLR(optim_D,
                                            step_size=args.decay_every,
                                            gamma=args.lr_decay)

    #----------------------------------------------------------------------
    # Load checkpoint and evaluate the model
    #----------------------------------------------------------------------
    if args.load:
        # In case we load model to contrinue from last epoch
        if args.best:
            ckpt_path_load = ckpt_path + "_best"
        elif args.batch:
            ckpt_path_load = ckpt_path + "_batch"
        else:
            ckpt_path_load = ckpt_path
        epoch_st, best_loss = utils.load_model(ckpt_path_load, model, device)
        epoch_st += 1
        print("Current best loss: %.8f" % best_loss)
    else:
        epoch_st, best_loss = 1, np.infty

    if args.load_segm_path:
        _, _ = utils.load_model(args.load_segm_path,
                                model,
                                device,
                                layers=['segm'])

    #-----------------------------------------------------------------------
    # Data definition
    #-----------------------------------------------------------------------
    # data_path is exported from config.py
    utils.set_seed(0)
    if args.data == 'synthetic' or args.normalize_method == "mean":
        print("No noise is removed")
        remove_noise_s = 0
        remove_noise_t = 0
    else:
        # Because for some source images we have too much of the smallest value,
        # while target's similar region is very little,
        # we makes smallest region on target bigger, to easily find warp.
        # Note, it does not effect the theoretical properties of the RF,
        # because in test we care about maximum, not minimum.
        remove_noise_s = 0.05
        remove_noise_t = 0.1

    source_loader, sample_size_source, is_mask, base_loader = utils.create_dataloader(
        path_data=data_path['source']['data'],
        path_mask=data_path['source']['mask'],
        path_base=data_path['source']['base'],
        batch_size=args.batch_size,
        is_normalize=True,
        n_combine=args.n_combine,
        shuffle=args.shuffle,
        add_noise=False,
        path_noise=data_path['target']['data'],
        remove_noise=remove_noise_s,
        log="depth" in args.data,
        normalize_method=args.normalize_method)

    if args.segm_only:
        target_test_loader, sample_size_test_target, _, _ = utils.create_dataloader(
            path_data=data_path['target']['data'],
            path_mask=data_path['target']['mask'],
            path_base=None,
            batch_size=args.batch_size,
            is_normalize=True,
            n_combine=args.n_combine,
            shuffle=args.shuffle,
            remove_noise=remove_noise_t,
            normalize_method=args.normalize_method,
            mode='test')
        target_test_loader = utils.inf_generator(target_test_loader)
        mode = 'train'
    else:
        mode = 'full'

    target_loader, sample_size_target, _, _ = utils.create_dataloader(
        path_data=data_path['target']['data'],
        path_mask=data_path['target']['mask'],
        path_base=None,
        batch_size=args.batch_size,
        is_normalize=True,
        n_combine=args.n_combine,
        shuffle=args.shuffle,
        remove_noise=remove_noise_t,
        normalize_method=args.normalize_method,
        mode=mode)

    source_loader = utils.inf_generator(source_loader)
    target_loader = utils.inf_generator(target_loader)

    if base_loader is not None:
        base_loader = utils.inf_generator(base_loader)

    # During training we would like to exploit all sampels of source,
    # which might be much larger than number of samples in target.
    # However, during final testing we use source number of samples only
    if args.test_only:
        sample_size = sample_size_source
    else:
        sample_size = max(sample_size_source, sample_size_target)

    args.n_epochs_to_viz = max(args.freq_gen_update, args.freq_disc_update)

    #-----------------------------------------------------------------------
    # Training
    log_path = "logs/" + str(experimentID) + ".log"
    os.makedirs("logs/", exist_ok=True)
    sys.stdout = Logger(log_path)
    #args.batch_size = min(args.batch_size, sample_size)
    n_batches = max(sample_size // args.batch_size, 1)

    os.makedirs("logs/tb/", exist_ok=True)
    tb_writer = SummaryWriter('logs/tb/%s' % (experimentID))

    if args.test_only:
        print("Testing results achieved at epoch %d" % epoch_st)
        with torch.no_grad():
            print("-----------------")
            # we want to see results on the target data only for segmentation
            if not args.segm_only:
                print("Testing source:")
                test(
                    args,
                    source_loader,
                    model,
                    epoch=0,
                    n_batches=10,  #n_batches,
                    compute_summary=True,
                    make_plots=True,
                    max_n_plots=args.n_plots,
                    best_loss=None,
                    plots_path=args.plots_path + "/source/",
                    ckpt_path=ckpt_path,
                    base_loader=base_loader,
                    is_mask=is_mask,
                    is_source=True,
                    synthetic_path="./tmp",
                    testloader=target_loader)
            else:
                print("Testing target:")
                test(args,
                     target_loader,
                     model,
                     epoch=0,
                     n_batches=n_batches,
                     compute_summary=True,
                     make_plots=True,
                     max_n_plots=args.n_plots,
                     best_loss=None,
                     plots_path=args.plots_path + "/target/",
                     ckpt_path=ckpt_path,
                     base_loader=None,
                     is_mask=True,
                     is_source=False,
                     synthetic_path="./tmp")
    else:
        for epoch in range(epoch_st):
            # In case we load model and need to update schedulers to
            # appropriate state
            scheduler_G.step()
            scheduler_D.step()

        for epoch in range(epoch_st, args.n_epochs + 1):
            print("lr_G: ", optim_G.param_groups[0]['lr'])
            print("lr_D: ", optim_D.param_groups[0]['lr'])
            print('Epoch %04d' % epoch)
            best_loss = train(args,
                              source_loader,
                              target_loader,
                              model,
                              optims,
                              n_batches=n_batches,
                              epoch=epoch,
                              ckpt_path=ckpt_path,
                              is_mask=is_mask,
                              best_loss=best_loss,
                              tb_writer=tb_writer,
                              train_gen=epoch % args.freq_gen_update == 0,
                              train_disc=epoch % args.freq_disc_update == 0,
                              train_reverse=epoch % args.freq_rec_update == 0)

            scheduler_G.step()
            scheduler_D.step()

            # Do testing and report summary
            if epoch % args.n_epochs_to_viz == 0 and\
               epoch >= args.n_epochs_start_viz:
                with torch.no_grad():
                    print("-----------------")

                    if args.segm_only:
                        best_loss = test(
                            args,
                            target_test_loader,
                            model,
                            epoch=epoch,
                            n_batches=max(
                                sample_size_test_target // args.batch_size,
                                1),  #n_batches,
                            compute_summary=True,
                            make_plots=False,
                            best_loss=best_loss,
                            ckpt_path=ckpt_path,
                            is_mask=True,
                            is_source=False)

                    elif args.method == 'within':
                        # If within we save only models which
                        # reject at least args.pval_targ on testing target
                        # Ideally it should be 100% of rejection, since we map
                        # one rejection map to another.
                        # It is implemented in source testing through
                        # condition = -test_loss >= args.pval_targ
                        # print("Testing target:")
                        # test_loss = test(args,
                        #                  target_loader,
                        #                  model,
                        #                  epoch=epoch,
                        #                  n_batches=n_batches,
                        #                  compute_summary=True,
                        #                  make_plots=False,
                        #                  best_loss=None,
                        #                  ckpt_path=ckpt_path,
                        #                  is_mask=True,
                        #                  is_source=False)
                        #test_loss = 1 # to ignore test
                        test_loss = -1

                        print("test_loss", test_loss)
                        if -test_loss >= args.pval_targ:
                            print("Testing source:")
                            best_loss = test(
                                args,
                                source_loader,
                                model,
                                epoch=epoch,
                                n_batches=n_batches,  #// 20 + 1,
                                compute_summary=True,
                                make_plots=False,
                                best_loss=best_loss,
                                ckpt_path=ckpt_path,
                                is_mask=True,
                                condition=-test_loss >= args.pval_targ,
                                is_source=True)

                    else:
                        print("Testing source:")
                        _ = test(args,
                                 source_loader,
                                 model,
                                 epoch=epoch,
                                 n_batches=n_batches,
                                 compute_summary=True,
                                 make_plots=False,
                                 best_loss=None,
                                 ckpt_path=ckpt_path,
                                 is_mask=True,
                                 is_source=True)

                        ## Just for chi-squared experiments
                        print("Testing target:")
                        best_loss = test(
                            args,
                            target_loader,
                            model,
                            epoch=epoch,
                            n_batches=n_batches,
                            compute_summary=True,
                            make_plots=False,
                            best_loss=best_loss,
                            ckpt_path=ckpt_path,
                            condition=True,  #always save if loss < best_loss
                            is_mask=is_mask,
                            is_source=False)


if __name__ == '__main__':
    main()
