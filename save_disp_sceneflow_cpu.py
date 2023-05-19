from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.utils as vutils
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
import gc
import skimage
import skimage.io
import cv2

parser = argparse.ArgumentParser(description='Attention Concatenation Volume for Accurate and Efficient Stereo Matching (ACVNet)')
parser.add_argument('--model', default='acvnet', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--dataset', default='sceneflow', help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', default="./Sampler", help='data path')
parser.add_argument('--testlist', default='./sceneflow_test.txt', help='testing list')
parser.add_argument('--loadckpt', default='./sceneflow_small.ckpt', help='load the weights from a specific checkpoint')
# parse arguments
args = parser.parse_args()

# dataset, dataloader
StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=4, drop_last=False)

# model, optimizer
model = __models__[args.model](args.maxdisp, False, False)
model = nn.DataParallel(model)
model.cpu()

# load parameters
print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt, map_location=torch.device('cpu'))
# Remove the "module." prefix from state_dict keys
state_dict = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
model.load_state_dict(state_dict)

save_dir = './'


def test():
    os.makedirs(save_dir, exist_ok=True)
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        disp_est_np = tensor2numpy(test_sample(sample))
        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader), time.time() - start_time))
        left_filenames = sample["left_filename"]

        for disp_est, fn in zip(disp_est_np, left_filenames):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est, dtype=np.float32)
            fn = os.path.join(save_dir, fn.split('/')[-4]+fn.split('/')[-3]+fn.split('/')[-1])
            print("saving to", fn, disp_est.shape)
            disp_est_uint = np.round(disp_est * 256).astype(np.uint16)
            cv2.imwrite(fn, cv2.applyColorMap(cv2.convertScaleAbs(disp_est_uint, alpha=0.01), cv2.COLORMAP_JET))


# test one sample
@make_nograd_func
def test_sample(sample):
    model.eval()
    disp_ests = model(sample['left'], sample['right'])
    return disp_ests[-1]


if __name__ == '__main__':
    test()

