import gc

import torch
from torch.utils.data import DataLoader

from architectures import RCAN
from create_image import create_image
from dataLoader import Loader
from training import epoch_train
from validation import validate

device = torch.device("cuda:0")


trainLoader = DataLoader(Loader("train", 2, 1000, make_crop=True,
                                    crop_size=150), batch_size=8)
testLoader = DataLoader(Loader("test", 2, 100, make_crop=True,
                                    crop_size=200), batch_size=8)
net = RCAN(64, 10, 20).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 [20, 30, 40], 0.2)
lossfunc = torch.nn.L1Loss().to(device)

epochs = 50
for i in range(epochs):
    print(f"Epoch {i+1}: ", end = ' ')
    trainloss = epoch_train(net, trainLoader, lossfunc, optimizer)
    valloss, psnr = validate(net, testLoader, lossfunc)
    scheduler.step()
    print(f"Train loss: {trainloss}\nVal loss: {valloss}\nPSNR: {psnr}\n")
torch.save(net.state_dict(), 'mrcan_parameters.pth')