import torch
from torch import nn

from metric.fid_score import _compute_statistics_of_ims, calculate_frechet_distance
from metric.inception import InceptionV3
from utils import util


def get_fid(fakes, model, npz, device, batch_size=1, tqdm_position=None):
    m1, s1 = npz['mu'], npz['sigma']
    fakes = torch.cat(fakes, dim=0)
    fakes = util.tensor2im(fakes).astype(float)
    m2, s2 = _compute_statistics_of_ims(fakes, model, batch_size, 2048,
                                        device, tqdm_position=tqdm_position)
    return calculate_frechet_distance(m1, s1, m2, s2)

def create_metric_models(opt, device):
    if not opt.no_fid:
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        inception_model = InceptionV3([block_idx])
        if len(opt.gpu_ids) > 1:
            inception_model = nn.DataParallel(inception_model, opt.gpu_ids)
        inception_model.to(device)
        inception_model.eval()
    else:
        inception_model = None
    return inception_model
