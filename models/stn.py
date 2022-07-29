import torch
import torch.nn as nn
import torch.nn.functional as nnf


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """
    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid, persistent=False)

    def forward(self, src, flow, grid=None, return_grid=False):

        # Normal grid
        shape = flow.shape[2:]
        new_locs = self.grid + flow
        # need to normalize grid values to [-1, 1] for resampler
        min_max = []
        if return_grid:
            grid = new_locs.clone()  # to return, used to plot reverse grid
        for i in range(len(shape)):
            # Corresponds to clipping > shape and < 0
            #new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] - -2) / (2 - -2) - 1
            #new_locs[:, i, ...] = 2 * new_locs[:, i, ...] / (shape[i] - 1) - 1 #usual
            ##new_locs[:, i, ...] = new_locs[:, i, ...] / (shape[i] - 1)

            # Correspond to scale from -1 to 1
            min_x = torch.amin(new_locs[:, i],
                               axis=[1, 2]).view(flow.shape[0], 1,
                                                 1).expand((-1, ) + shape)
            max_x = torch.amax(new_locs[:, i],
                               axis=[1, 2]).view(flow.shape[0], 1,
                                                 1).expand((-1, ) + shape)
            min_max.append(2 * (new_locs[:, i] - min_x) / (max_x - min_x) - 1)

        new_locs = torch.stack(min_max, axis=1)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        ans = nnf.grid_sample(src,
                              new_locs,
                              align_corners=True,
                              mode=self.mode,
                              padding_mode="border")

        if return_grid:
            return ans, grid
        else:
            return ans
