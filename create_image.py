import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage as ToImage
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw

from dataLoader import Loader
from validation import compute_metrics


def create_image(net, loader, name, upscaling=2, multiscale=False):
    patch_size = 30
    patches = np.array(
        [[85, 90, 85 + patch_size, 90 + patch_size],
         [160, 140, 160 + patch_size, 140 + patch_size],
         [350, 80, 350 + patch_size, 80 + patch_size]])

    fig = plt.figure(figsize=(15, 18), dpi=100)
    gs = fig.add_gridspec(4, 3, wspace=0.01, hspace=0.3, left=0.05,
                          top=0.95, bottom=0.02, right=0.95)

    highres, lowres = loader[0]
    lowres = lowres.to(torch.device("cuda:0"))
    lowres = lowres.view([1] + list(lowres.size()))
    net.eval()
    with torch.no_grad():
        if (multiscale):
            superres = net(lowres, upscaling)
        else:
            superres = net(lowres)
    lowres = ToImage()(lowres.cpu().view(lowres.size()[1:]))
    superres = ToImage()(superres.cpu().view(superres.size()[1:]))
    highres = ToImage()(highres.cpu().view(highres.size()[1:]))

    lowres_draw = lowres.copy()
    draw = ImageDraw.Draw(lowres_draw)
    draw.rectangle(list(patches[0]), outline='white')
    draw.rectangle(list(patches[1]), outline='white')
    draw.rectangle(list(patches[2]), outline='white')

    lowres = lowres.resize((lowres.size[0] * upscaling,
                            lowres.size[1] * upscaling),
                           Image.BICUBIC)

    psnr_low, psnr_super, ssim_low, ssim_super = compute_metrics(
        net, DataLoader(loader), upscaling, multiscale)
    lowres_title = "Low-resolution image" +\
    "\nAverage PSNR over the dataset: {:.2f}\n".format(psnr_low) +\
    "Average SSIM over the dataset: {:.4f}".format(ssim_low)
    superres_title = "Reconstructed image\n" +\
    "Average PSNR over the dataset: {:.2f}\n".format(psnr_super) +\
    "Average SSIM over the dataset: {:.4f}".format(ssim_super)
    fig.add_subplot(gs[0, 0], xticks=[], yticks=[],
                    ylabel=f"Image", title=lowres_title)
    plt.imshow(np.array(lowres_draw))
    fig.add_subplot(gs[0, 1], xticks=[], yticks=[],
                    title=superres_title)
    plt.imshow(np.array(superres))
    fig.add_subplot(gs[0, 2], xticks=[], yticks=[],
                    title="High-resolution image")
    plt.imshow(np.array(highres))

    ylabels = ["Patch 1", "Patch 2", "Patch 3"]
    for i in range(3):
        print(lowres.size, highres.size, superres.size)
        lowres_patch = lowres.crop(patches[i] * upscaling)
        highres_patch = highres.crop(patches[i] * upscaling)
        superres_patch = superres.crop(patches[i] * upscaling)

        fig.add_subplot(gs[1 + i, 0], xticks=[], yticks=[],
                        ylabel=ylabels[i])
        plt.imshow(np.array(lowres_patch))
        fig.add_subplot(gs[1 + i, 1], xticks=[], yticks=[])
        plt.imshow(np.array(superres_patch))
        fig.add_subplot(gs[1 + i, 2], xticks=[], yticks=[])
        plt.imshow(np.array(highres_patch))

    plt.savefig(name)