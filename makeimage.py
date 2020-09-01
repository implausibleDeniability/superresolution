import torch
from torch.utils.data import DataLoader

from architectures import RCAN
from create_image import create_image
from dataLoader import Loader

net = RCAN(64, 10, 20)
net.load_state_dict(torch.load("div2k/parameters/rcan_parameters.pth"))
net.to(torch.device("cuda:0"))

loader = Loader('val', scale=2, make_crop=False)
create_image(net, loader, "mrcan_result", upscaling=2, multiscale=False)
