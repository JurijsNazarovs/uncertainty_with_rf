import torch.nn as nn
import math
import os
import pickle
import numpy as np
import torch
from sklearn import metrics

import rf_collection as rfc
from torch.utils.data import DataLoader, TensorDataset
import cv2
#from skimage.transform import rotate
import torchvision.transforms as transforms
import random
import torch.nn.functional as F


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def combine_batch(x, fac=1, imgw=1216):
    #fac is how many images to comnbine

    if torch.is_tensor(x):
        x = torch.split(x, fac, 0)  #split on batches of images to combine
        x = torch.stack(
            x, axis=0)  #now, each x[i] have to be combined with proper shape
        x = torch.split(x, imgw // x.shape[-1], 1)
        x = torch.stack(x, axis=0)
        #x = x.permute(1, 3, 0, 4, 2, 5)
        # import pdb
        # pdb.set_trace()

        x = x.permute(1, 3, 0, 4, 2, 5)
        #x = x.permute(1, 3, 0, 2, 4, 5) #bad
        #x = x.permute(1, 3, 4, 0, 2, 5) #bad

        # import pdb
        # pdb.set_trace()
        # x = torch.cat(x, axis=0)

        # dims = tuple(range(len(x.shape)))
        # #x = x.permute(0, 2, 3, 1, 4)
        # #x = x.permute((0, ) + dims[2:-1] + (1, dims[-1]))
        # x = x.permute(0, 2, 3, 4, 1)
    else:
        x = np.split(x, fac, 0)
        x = np.stack(x, axis=0)

        dims = tuple(range(len(x.shape)))
        x = np.transpose(x, (0, ) + dims[2:-1] + (1, dims[-1]))
    #x = x.reshape(x.shape[:-2] + (-1, ))

    x = x.reshape(x.shape[:2] + (-1, imgw))

    #x = x.reshape(1, 1, -1, fac)
    return x


def normalize(x, q=None, method='scale', xmin=None, xmax=None):
    # Important for classification.
    # x_max = torch.amax(x, (2, 3), keepdim=True)
    # x_min = torch.amin(x, (2, 3), keepdim=True)
    # x = (x - x_min) / (x_max - x_min)

    # 0, 1 scalling
    #zmin, zmax = -3, 3
    #xmin, xmax = x.min(), x.max()
    # if torch.is_tensor(x):
    #     xmin, xmax = torch.quantile(x, 0), torch.quantile(x, q)
    # else:
    #     xmin, xmax = np.quantile(x, 0), np.quantile(x, q)
    #x = (x - xmin) / (xmax - xmin) * (zmax - zmin) + zmin
    # #x = (x - x.mean())

    # mu/sigma standartization. If q is provided, compute mu, sigma on those only

    if method == 'mean':
        if q is not None:
            max_per_x = x.amax(dim=[2, 3])
            if torch.is_tensor(max_per_x):
                ind = max_per_x <= torch.quantile(max_per_x, q)
            else:
                ind = max_per_x <= np.quantile(max_per_x, q)
            ind = ind.reshape(-1, x.shape[1])  #(-1)
            x_ = x[ind]
        else:
            x_ = x
        mu = x_.mean()  #dim=0)
        std = x_.std()  #dim=0)
        print("mu: %f, std: %f" % (mu, std))
        print("Before norm: ", x.min(), x.max())
        x = (x - mu) / std
        print("After norm: ", x.min(), x.max())
        mu = x.mean()  #dim=0)
        std = x.std()  #dim=0)
        print("After norm mu: ", mu.min(), mu.max())
        print("After norm std: ", std.min(), std.max())

    elif method == 'scale':
        if xmin is None or xmax is None:
            #xmin, xmax = x.min(), x.max()

            # Before it was (2,3) for imgs with 1 channel
            xmax = torch.amax(x, (1, 2, 3), keepdim=True)  #.detach()
            xmin = torch.amin(x, (1, 2, 3), keepdim=True)  #.detach()

            #print(xmin, xmax)

        #print(xmax)
        #print(xmin)
        x = (x - xmin) / (xmax - xmin)  #* 2 - 1  #* 6 - 3
    elif method == 'sigmoid':
        if torch.is_tensor(x):
            x = torch.nn.functional.sigmoid(x)
    elif method == 'clip' and q is not None:
        #max_per_x = x.amax(dim=[2,3])
        clipval = torch.quantile(x.reshape(x.shape[0], -1), q, dim=1)
        for i in range(len(clipval)):
            x[i] = torch.clip(x[i], max=clipval[i])
    else:
        raise ValueError("Unknown normalization method:", method)

    return x


def save_model(args, model, ckpt_path, epoch=0, best_loss=np.infty):
    generator = model.gen
    discriminator = model.discr
    classifier = model.segm

    torch.save(
        {
            'args': args,
            'state_dict_gen': generator.state_dict(),
            'state_dict_discr': discriminator.state_dict(),
            'state_dict_segm': classifier.state_dict(),
            # 'state_dict_gen2': model.gen2.state_dict(),
            # 'state_dict_discr2': model.discr2.state_dict(),
            # 'state_dict_stn': model.stn.state_dict(),
            'epoch': epoch,
            'best_loss': best_loss,
        },
        ckpt_path)


def load_model(ckpt_path, model, device, layers=['gen', 'discr']):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    else:
        print("Loading model from %s" % ckpt_path)
    # Load checkpoint.
    checkpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = checkpt['args']  # Not used?
    epoch_st = checkpt['epoch']
    best_loss = checkpt['best_loss']
    #if 'best_loss' in checkpt.keys() else np.infty

    for layer in layers:
        state_dict = checkpt['state_dict_' + layer]
        _model = getattr(model, layer)
        model_dict = _model.state_dict()
        # 1. filter out unnecessary keys
        state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        # if layer == "gen":
        #     state_dict['jac_weight'] = torch.ones(1)
        #     state_dict['outgrid_weight'] = torch.ones(1)

        # 2. overwrite entries in the existing state dict
        model_dict.update(state_dict)
        # 3. load the new state dict
        _model.load_state_dict(state_dict)
        _model.to(device)
    #del checkpt
    return epoch_st, best_loss


def iou_numpy(outputs: np.array, labels: np.array):
    SMOOTH = 1e-6

    outputps = outputs.astype(int)
    labels = labels.astype(int)

    intersection = (outputs & labels).sum()
    union = (outputs | labels).sum()

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    #thresholded = np.ceil(np.clip(20 * (iou - 0.5), 0, 10)) / 10
    return iou  #thresholded  # Or thresholded.mean()


def get_binary_summary(x, y, old_summary=None, n_batches=1, method="method"):
    '''
    Function contains summary statistics for binary classification, 
    to evaluate out segmentation
    x, y - different masks
    '''
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(y, torch.Tensor):
        y = y.detach().cpu().numpy()

    # if len(x.shape) > 2:
    #     x = x.reshape(x.shape[0], -1)
    # if len(y.shape) > 2:
    #     y = y.reshape(y.shape[0], -1)

    pval_true = np.mean((x == 1).any(axis=1))
    pval_model = np.mean((y == 1).any(axis=1))

    #power_cross_rf = np.mean((x == 1).any(axis=1) == (y == 1).any(axis=1))

    sig_x = np.where((x == 1).any(axis=1))[0]
    sig_y = np.where((y == 1).any(axis=1))[0]
    sig_x_in_sig_y = np.where(np.in1d(sig_x, sig_y))[0]
    if len(sig_x) == 0:
        power_cross_rf = 0
    else:
        power_cross_rf = len(sig_x_in_sig_y) / len(sig_x)

    x = x.reshape(-1)
    y = y.reshape(-1)

    conf_mat = metrics.confusion_matrix(x, y, labels=[0,
                                                      1])  #, normalize='true')
    report = metrics.classification_report(x,
                                           y,
                                           output_dict=True,
                                           zero_division=0,
                                           labels=[0, 1])
    summary = {}
    summary['conf_mat'] = np.around(conf_mat, 4)
    #dict_keys(['0', '1', 'accuracy', 'macro avg', 'weighted avg'])
    summary.update(report['1'])
    summary['power'] = report['1']['recall']
    summary['alpha'] = 1 - report['0']['recall']

    summary['iou'] = iou_numpy(x, y)
    summary['pval_true'] = pval_true
    summary['pval_model'] = pval_model
    summary['power_cross_rf'] = power_cross_rf

    if old_summary is not None:
        if method not in old_summary.keys():
            old_summary[method] = {}

        for key, item in summary.items():
            if key in old_summary[method].keys():
                old_summary[method][key] += item / n_batches
            else:
                old_summary[method][key] = item / n_batches
        return old_summary
    else:
        return summary


def make_plots(img,
               mask_hat,
               mask=None,
               img_fake=None,
               plots_path='',
               mask_cochran=None,
               mask_quantile=None,
               base_img=None):
    os.makedirs(os.path.dirname(plots_path), exist_ok=True)
    #mask_hat = torch.bitwise_and(
    #    mask_hat == 1, img > torch.quantile(img, 0)).to(torch.int64)  #0.8

    rfc.plot_rf(img, plot_path="%s_rf.png" % plots_path)
    rfc.plot_rf(mask_hat,
                plot_path="%s_mask_hat.png" % plots_path,
                is_mask=True)
    rfc.plot_rf(img,
                mask=mask_hat,
                plot_path="%s_rf_mask_hat.png" % plots_path)

    if mask is not None:
        rfc.plot_rf(-mask, plot_path="%s_mask.png" % plots_path, is_mask=True)
        overlap = torch.bitwise_and(mask_hat == 1, mask == 1)
        diff = (mask_hat - mask).to(torch.float)
        diff[overlap] += 0.3
        rfc.plot_rf(diff,
                    plot_path="%s_mask_hat_vs_true.png" % plots_path,
                    is_mask=True)

    if img_fake is not None:
        # Fake images: make sense for source input
        rfc.plot_rf(img_fake, plot_path="%s_fake.png" % plots_path)

    if mask_cochran is not None:
        rfc.plot_rf(mask_cochran,
                    plot_path="%s_mask_cochran.png" % plots_path,
                    is_mask=True)
        if mask is not None:
            rfc.plot_rf(mask_cochran - mask,
                        plot_path="%s_mask_cochran_vs_true.png" % plots_path,
                        is_mask=True)

    if mask_quantile is not None:
        rfc.plot_rf(mask_quantile,
                    plot_path="%s_mask_q.png" % plots_path,
                    is_mask=True)
        if mask is not None:
            overlap = torch.bitwise_and(mask_quantile == 1, mask == 1)
            diff = (mask_quantile - mask).to(torch.float)
            diff[overlap] += 0.3
            rfc.plot_rf(diff,
                        plot_path="%s_mask_q_vs_true.png" % plots_path,
                        is_mask=True)

    if base_img is not None:
        # plot mask above base_img
        print(base_img.shape)
        if base_img.shape[0] == 3:
            #base_img = base_img.permute(1, 2, 0).type(torch.int64)
            base_img = base_img.data.cpu().numpy()
            base_img = np.transpose(base_img, (1, 2, 0))
            if base_img.max() > 2:
                base_img = base_img / 255

            rfc.plot_rf(
                base_img,  #.astype(np.uint8),
                plot_path="%s_base.png" % plots_path)
            #base_img = cv2.cvtColor(base_img,
            #                        cv2.COLOR_BGR2GRAY)  #.astype(np.uint8)
            print(base_img.shape)

        rfc.plot_rf(base_img,
                    mask=mask_hat,
                    plot_path="%s_base_h.png" % plots_path,
                    base_cmap='Greys')

        if mask_quantile is not None:
            rfc.plot_rf(base_img,
                        mask=mask_quantile,
                        plot_path="%s_base_q.png" % plots_path,
                        base_cmap='Greys')


# ------------------------------------------------------------------------------
# Data loaders
# ------------------------------------------------------------------------------


def get_data(path, device="cpu"):
    """
    Function to load random fields from directory
    """
    if os.path.isdir(path):
        data = []
        for f in os.listdir(path):

            if not os.path.isfile("%s/%s" % (path, f)):
                continue
            batch = pickle.load(open("%s/%s" % (path, f), "rb"))

            data.append(batch)

        data = np.concatenate(data)
        print("Read data from the %s of the following shape %s" %
              (path, data.shape))
    else:
        data = pickle.load(open(path, "rb"))
    #return torch.tensor(data[:10 * 95], dtype=torch.float32).to(device)
    return torch.tensor(data, dtype=torch.float32).to(device)


# class Data():
#     def __init__(self,
#                  path_s,
#                  path_t,
#                  path_mask_s=None,
#                  path_mask_t=None,
#                  device='cpu',
#                  scale=10,
#                  alpha=0.05,
#                  L=32):
#         self.source = get_data(path_s, device)
#         self.target = get_data(path_t, device)
#         if path_mask_s is None:
#             # Generate source map based on Gaussian RF
#             source_map, _, _ = rfc.get_significant(self.source.cpu().numpy(),
#                                                    test='rf',
#                                                    rf_type='gauss',
#                                                    alpha=alpha,
#                                                    scale=scale,
#                                                    L=L)
#             self.source_map = torch.tensor(source_map).to(self.source)
#         else:
#             self.source_map = get_data(path_mask_s, device)

#         if path_mask_t is None:
#             self.target_mask = None
#         else:
#             self.target_mask = get_data(path_mask_t, device)

#         # assert(len(self.source) = len(self.mask_source)),\
#         #     "For source mask and data are not of the same size"

#     # ------------------------------
#     # Dataloader if we have access to data as array
#     # ------------------------------
#     def __len__(self):
#         return len(self.source)

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx,
#                                                                          0])
#         image = io.imread(img_name)
#         landmarks = self.landmarks_frame.iloc[idx, 1:]
#         landmarks = np.array([landmarks])
#         landmarks = landmarks.astype('float').reshape(-1, 2)
#         sample = {'image': image, 'landmarks': landmarks}

#         if self.transform:
#             sample = self.transform(sample)

#         return sample

#     # ------------------------------
#     # Data loader if we read files
#     # ------------------------------
#     # def __len__(self):
#     #     return self.tot_samples

#     # def __iter__(self):
#     #     self.batch_idx = 0
#     #     return self

#     # def __next__(self):
#     #     # if self.n <= self.max:
#     #     #     result = 2 ** self.n
#     #     #     self.n += 1
#     #     #     return result
#     #     # else:
#     #     #     raise StopIteration
#     #     batch = self._getbatch(self.batch_idx)
#     #     if batch.shape[0] == 0:
#     #         raise StopIteration
#     #     else:
#     #         self.batch_idx += 1
#     #     return batch

#     # def _getbatch(self, batch_idx):
#     #     if self.is_train:
#     #         inds = self.train_idxs
#     #     else:
#     #         inds = self.test_idxs
#     #     start_idx = batch_idx * self.batch_size
#     #     end_idx = min(start_idx + self.batch_size, len(inds))

#     #     fnames = [
#     #         "%s/%04d.npy" % (self.datapath, idx)
#     #         for idx in inds[start_idx:end_idx]
#     #     ]
#     #     X_batch = self.load_data(fnames=fnames)
#     #     if self.device is not None:
#     #         X_batch = torch.tensor(X_batch,
#     #                                requires_grad=False).to(self.device)

#     #     #return X_batch[:, :-1, :, :, :], X_batch[:, -1, :, :, :] #x, y
#     #     #import pdb
#     #     #pdb.set_trace()
#     #     #hui = torch.unbind(X_batch, 0)
#     #     return X_batch


def create_dataloader(path_data,
                      path_mask=None,
                      path_base=None,
                      batch_size=1,
                      is_normalize=False,
                      n_combine=4,
                      shuffle=True,
                      add_noise=False,
                      path_noise=None,
                      remove_noise=0,
                      log=False,
                      normalize_method='scale',
                      mode='full'):
    # mode: 'full' - full data is used, 'train'- 0.7 left used,
    # 'test' - 0.3 right used
    # path_base is image we want to put mask on
    device = 'cpu'  # to make sure that GPU is not occupied with all data
    data = get_data(path_data, device)

    if is_normalize:
        # # Clean data from outliers: remove images which have values higher
        # # than q quantile.
        # q = 0.99
        # max_per_x = data.amax(dim=[2, 3])  #max per image
        # if n_combine > 1:
        #     ind = np.where(
        #         (max_per_x > np.quantile(max_per_x, q)).reshape(-1))[0]

        #     ind = [
        #         x for i in ind for x in range(i - (i % n_combine), i +
        #                                       (n_combine - i % n_combine))
        #     ]
        #     ind = [i for i in range(len(data)) if i not in ind]

        # else:
        #     ind = max_per_x <= np.quantile(max_per_x, q)
        #     ind = ind.reshape(-1)

        # # # #ind = list(range(10**2 - 10))
        # # # print(data.shape)

        #ind = range((min(100 * 4, len(data))))
        #ind = range(min(4 * 1000, len(data)))

        #ind = range(min(32 * 32, len(data)))
        ind = range(len(data))
        #ind = range(10**3)

        train_frac = 0.7
        if mode == 'train':
            ind = range(int(len(ind) * train_frac))
        elif mode == 'test':
            ind = range(int(len(ind) * train_frac), len(ind))
        else:
            print("The full dataset is used")

        data = data[ind]  #filtering data

        # Clip data by quantile? necesssary for kitty. some sample are too big
        if log:
            data = normalize(data, q=0.99, method='clip')
            data = torch.log(data)
        data = normalize(data, q=1, method=normalize_method)  #default is scale
        if add_noise:
            ind_noise = data < 0.2
            if ind_noise.sum() > 0:
                #noise = torch.abs(torch.randn(data.shape))
                if path_noise is None:
                    #noise = torch.randn(data.shape)
                    noise = torch.rand(data.shape)
                else:
                    noise = get_data(path_noise, device)[ind]
                noise = normalize(noise, q=1, method='scale')
                noise = torch.clip(noise, 0, 0.6)
                data[ind_noise] = noise[ind_noise]
        if remove_noise > 0:
            ind_noise = data < remove_noise
            if ind_noise.sum() > 0:
                data[ind_noise] = 0

        #######
        # data[data > 0.9] = 0
        #########

    seed = random.randint(0, 2**24)
    set_seed(seed)  # to shuffle base in a right way

    def _init_fn(worker_id):
        torch.initial_seed()
        #np.random.seed(seed)

    batch_size = min(data.shape[0], batch_size)
    if path_mask is None:
        loader = DataLoader(data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            worker_init_fn=_init_fn,
                            num_workers=0,
                            drop_last=True)
        is_mask = False
    else:
        mask = get_data(path_mask, device).to(torch.int64).squeeze()  #CE loss
        #mask = get_data(path_mask, device).to(torch.float32).squeeze()  #BCE
        if is_normalize:
            mask = mask[ind]

        loader = DataLoader(TensorDataset(data, mask),
                            batch_size=batch_size,
                            shuffle=shuffle,
                            worker_init_fn=_init_fn,
                            num_workers=0,
                            drop_last=True)
        is_mask = True

    if path_base is None:
        base = None
        base_loader = None
    else:
        base = get_data(path_base, device)
        if is_normalize:
            base = base[ind]

        set_seed(seed)
        base_loader = DataLoader(base,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 worker_init_fn=_init_fn,
                                 num_workers=0,
                                 drop_last=True)

    print(data.min())
    print(data.max())
    print(len(data))
    return loader, len(data), is_mask, base_loader


def inf_generator(iterable):
    """
    Allows training with DataLoaders in a single infinite loop:
    for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


import torch.autograd as autograd


def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA=10):
    #print real_data.size()
    BATCH_SIZE = real_data.shape[0]
    #LAMBDA = 100

    alpha = torch.rand(BATCH_SIZE, 1, 1, 1).to(real_data)
    alpha = alpha.repeat([1, 1] + list(real_data.shape[-2:]))
    #alpha = alpha.expand(real_data.size()).to(real_data)
    #alpha = alpha.cuda(gpu) if use_cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(
            disc_interpolates.size()).to(real_data),  #.cuda(gpu)
        #if use_cuda else torch.ones(disc_interpolates.size()),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    #gradient_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean() * LAMBDA
    gradient_penalty = ((1. - torch.sqrt(1e-8 + torch.sum(
        gradients.view(gradients.size(0), -1)**2, dim=1)))**2).mean() * LAMBDA
    if torch.isnan(gradient_penalty):
        import pdb
        pdb.set_trace()
        print("DEBUG! Disc loss is nan")
    return gradient_penalty


def init_network_weights(m):
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)


def get_jacdet2d_filter(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    # # Apply Sobell 3x3 filter
    # # Dx
    # a = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).to(displacement)
    # a = a.view((1, 1, 3, 3))

    # # Dy
    # b = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).to(displacement)
    # b = b.view((1, 1, 3, 3))

    # # Apply Robert 2x2 filter
    # # Dx
    # a = torch.Tensor([[1, 0], [0, -1]]).to(displacement)
    # a = a.view((1, 1, 2, 2))

    # # Dy
    # b = torch.Tensor([[0, 1], [-1, 0]]).to(displacement)
    # b = b.view((1, 1, 2, 2))

    # Apply average derivative 2x2 filter
    # Dx
    a = 1 / 2 * torch.Tensor([[-1, 1], [-1, 1]]).to(displacement)
    a = a.view((1, 1, 2, 2))

    # Dy
    b = 1 / 2 * torch.Tensor([[1, 1], [-1, -1]]).to(displacement)
    b = b.view((1, 1, 2, 2))

    # Take derivative
    Dx_x = F.conv2d(displacement[:, 0:1], a)
    Dx_y = F.conv2d(displacement[:, 0:1], b)

    Dy_x = F.conv2d(displacement[:, 1:], a)
    Dy_y = F.conv2d(displacement[:, 1:], b)

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    return jacdet


def get_jacdet2d(displacement, grid=None, backward=False):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*2*h*w
    '''
    Dx_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    Dx_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    Dy_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    Dy_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Dy_x = (displacement[:, 0, :-1, 1:] - displacement[:, 0, :-1, :-1])
    # Dy_y = (displacement[:, 0, 1:, :-1] - displacement[:, 0, :-1, :-1])
    # Dx_x = (displacement[:, 1, :-1, 1:] - displacement[:, 1, :-1, :-1])
    # Dx_y = (displacement[:, 1, 1:, :-1] - displacement[:, 1, :-1, :-1])

    # Normal grid
    if backward:
        D1 = (1 - Dx_x) * (1 - Dy_y)
    else:
        D1 = (1 + Dx_x) * (1 + Dy_y)
    #D1 = (Dx_x) * (Dy_y)
    D2 = Dx_y * Dy_x
    jacdet = D1 - D2

    # # tanh grid
    # grid_x = grid[:, 0, :-1, :-1]
    # grid_y = grid[:, 1, :-1, :-1]
    # coef = 1 - torch.tanh(torch.atanh(grid) + displacement)**2
    # coef_x = coef[:, 0, :-1, :-1]
    # coef_y = coef[:, 1, :-1, :-1]
    # D1 = (1 / (1 - grid_x**2) + Dx_x) * (1 / (1 - grid_y**2) + Dy_y)
    # D2 = Dx_y * Dy_x
    # jacdet = coef_x * coef_y * (D1 - D2)

    return jacdet


def get_jacdet3d(displacement):
    '''
    Calculate the Jacobian value at each point of the displacement map having
    size of b*h*w*d*3 and in the cubic volumn of [-1, 1]^3
    '''
    raise NotImplementedError("Need to fix")

    D_y = (displacement[:, 1:, :-1, :-1, :] -
           displacement[:, :-1, :-1, :-1, :])

    D_x = (displacement[:, :-1, 1:, :-1, :] -
           displacement[:, :-1, :-1, :-1, :])

    D_z = (displacement[:, :-1, :-1, 1:, :] -
           displacement[:, :-1, :-1, :-1, :])

    D1 = (D_x[..., 0] + 1) * ((D_y[..., 1] + 1) *
                              (D_z[..., 2] + 1) - D_z[..., 1] * D_y[..., 2])

    D2 = (D_x[..., 1]) * (D_y[..., 0] *
                          (D_z[..., 2] + 1) - D_y[..., 2] * D_x[..., 0])

    D3 = (D_x[..., 2]) * (D_y[..., 0] * D_z[..., 1] -
                          (D_y[..., 1] + 1) * D_z[..., 0])

    return D1 - D2 + D3


def jacdet_loss(vf, grid=None, backward=False):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    jacdet = get_jacdet2d(vf, grid, backward)
    ans = 1 / 2 * (torch.abs(jacdet) - jacdet).mean(axis=[1, 2]).sum()

    #ans = -torch.amin(jacdet, [1,2])
    #ans = torch.min(jacdet)
    return ans


def outgrid_loss(vf, grid, backward=False, size=32):  #32-1):
    '''
    Penalizing locations where Jacobian has negative determinants
    Add to final loss
    '''
    if backward:
        pos = grid - vf - (size - 1)
        neg = grid - vf
    else:
        pos = grid + vf - (size - 1)
        neg = grid + vf

    # penalize > size
    ans_p = 1 / 2 * (torch.abs(pos) + pos).mean(axis=[1, 2]).sum()
    # penalize < 0
    ans_n = 1 / 2 * (torch.abs(neg) - neg).mean(axis=[1, 2]).sum()
    ans = ans_n + ans_p

    return ans


def get_running(old_value, new_value, i):
    value = (old_value * i + new_value) / (i + 1)
    return value


def get_beta(beta_type="original",
             n_batches=1,
             batch_idx=1,
             reverse=False,
             weight=1):
    """
    Function returns beta for VI inference
    """

    if beta_type == "blundell":
        # https://arxiv.org/abs/1505.05424
        beta = 2**(n_batches - (batch_idx)) / (2**n_batches - 1)
    elif beta_type == "graves":
        # https://papers.nips.cc/paper/2011/file/7eb3c8be3d411e8ebfab08eba5f49632-Paper.pdf
        # eq (18)
        beta = 1 / n_batches
    elif beta_type == "cycle":
        beta = frange_cycle_linear(n_batches, n_cycle=1,
                                   ratio=0.5)[batch_idx] * 0.001
    elif beta_type == "default":
        beta = weight
    else:
        beta = 0

    if reverse:
        beta = 1 - beta
    return beta


def frange_cycle_linear(n_epoch, start=0., stop=1., n_cycle=4, ratio=0.5):
    # From here:  github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    # n_epochs can also be n_batches
    L = np.ones(n_epoch)
    period = n_epoch / n_cycle
    step = (stop - start) / (period * ratio)  # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_epoch):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def check_grads(model, model_name):
    grads = []
    for p in model.parameters():
        if not p.grad is None:
            grads.append(float(p.grad.mean()))

    grads = np.array(grads)
    if grads.any() and grads.mean() > 100:
        print("********\n"
              "WARNING! gradients mean is over 100", model_name, "\n********")
    if grads.any() and grads.max() > 100:
        print("********\n"
              "WARNING! gradients max is over 100", model_name, "\n********")


def aug_trans(data, mask=None):
    # if torch.is_tensor(data):
    #     data = data.numpy()

    transform = transforms.Compose([
        transforms.RandomAffine(degrees=180, translate=None),  #(0.1, 0.1)),
        #transforms.RandomCrop(32, padding=4),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomHorizontalFlip(p=0.2)
    ])
    if mask is not None:
        state = torch.get_rng_state()
        data = transform(data)
        torch.set_rng_state(state)
        mask = transform(mask)
        return (data, mask)
    else:
        return transform(data)


# python3 main.py --n_epochs 5000 --data fmnist --df 5 --experimentID fmnistlabel7_euler_lsgan_vxnet_du1_decay100_vf-init_img_scale_reclw-0.001_lrg0.0001_lrd0.0001_unet_lastwarp_removenoise_discupallways_simpledisc1batchlr_nosigm_jg0_dropoutdisc_shuffle_aug --device 1 --method within --batch_size 32 --normalize_gan --n_combine 1 --n_epochs_start_viz 50 --gen_loss_weight 1 --disc_loss_weight 1 --pval_targ 0.95 --gan_type lsgan --ode_solver euler  --gen_type vxnet --freq_gen_update 1 --decay_every 100 --lr_gen 0.0001 --lr_disc 0.0001 --ode_vf init_img_y --ode_norm scale --plots_path plots/  --rec_loss_weight 0.001 --freq_rec_update 1 --last_warp --jac_loss_weight_forw 0 --jac_loss_weight_back 0 --rec_weight_method default --outgrid_loss_weight_forw 1 --outgrid_loss_weight_back 1 --n_discblocks 4 --shuffle --augment --load

# Working non augmented, r_w=0.001, lg=0.0001, ld=0.0001, lsgan, shuffle, lastwarp:
# python3 main.py --n_epochs 5000 --data fmnist --df 5 --experimentID fmnistlabel7_euler_lsgan_vxnet_du1_decay100_vf-init_img_scale_reclw-0.001_lrg0.0001_lrd0.0001_unet_lastwarp_removenoise_discupallways_simpledisc1batchlr_nosigm_jg0_dropoutdisc_shuffle --device 0 --method within --batch_size 32 --normalize_gan --n_combine 1 --n_epochs_start_viz 50 --gen_loss_weight 1 --disc_loss_weight 1 --pval_targ 0.95 --gan_type lsgan --ode_solver euler  --gen_type vxnet --freq_gen_update 1 --decay_every 100 --lr_gen 0.0001 --lr_disc 0.0001 --ode_vf init_img_y --ode_norm scale --plots_path plots/  --rec_loss_weight 0.001 --freq_rec_update 1 --last_warp --jac_loss_weight_forw 0 --jac_loss_weight_back 0 --rec_weight_method default --outgrid_loss_weight_forw 1 --outgrid_loss_weight_back 1 --n_discblocks 4 --shuffle --load
