import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass


class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        self.fmap1 = fmap1
        self.fmap2 = fmap2

    def corr(self, fmap1, fmap2, coords):

        B, D, H, W = fmap2.shape
        fmap1 = fmap1.unsqueeze(dim=-1)
        fmap2 = fmap2.unsqueeze(dim=-1)

        # map grid coordinates to [-1,1]
        xgrid, ygrid = coords.split([1,1], dim=-1)
        xgrid = 2*xgrid/(W-1) - 1
        ygrid = 2*ygrid/(H-1) - 1
        zgrid = torch.zeros_like(xgrid) - 1
        grid = torch.cat([zgrid, xgrid, ygrid], dim=-1)

        fmapw = F.grid_sample(fmap2, grid, align_corners=True)

        corr = torch.sum(fmap1*fmapw, dim=1)
        return corr / torch.sqrt(torch.tensor(D).float())

    def __call__(self, coords):

        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        fmap1 = self.fmap1
        fmap2 = self.fmap2

        out_pyramid = []
        for i in range(self.num_levels):
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch, h1, w1, 1, 2) / 2**i
            coords_lvl = centroid_lvl + delta.view(-1, 2)

            corr = self.corr(fmap1, fmap2, coords_lvl)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

class CorrLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fmap1, fmap2, coords, r):
        fmap1 = fmap1.contiguous()
        fmap2 = fmap2.contiguous()
        coords = coords.contiguous()
        ctx.save_for_backward(fmap1, fmap2, coords)
        ctx.r = r
        corr, = correlation_cudaz.forward(fmap1, fmap2, coords, ctx.r)
        return corr

    @staticmethod
    def backward(ctx, grad_corr):
        fmap1, fmap2, coords = ctx.saved_tensors
        grad_corr = grad_corr.contiguous()
        fmap1_grad, fmap2_grad, coords_grad = \
            correlation_cudaz.backward(fmap1, fmap2, coords, grad_corr, ctx.r)
        return fmap1_grad, fmap2_grad, coords_grad, None


class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):

        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1)
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1)

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr = alt_cuda_corr(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / 16.0
