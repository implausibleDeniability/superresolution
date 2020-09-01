import torch
from torchvision.transforms import ToPILImage as ToImage
from torchvision.transforms import ToTensor
from pytorch_msssim import ms_ssim
from PIL import Image


def compute_psnr(image1, image2):
    mse = torch.nn.MSELoss()(image1, image2)
    return 10 / 2.30259 * torch.log(1 / mse)


def compute_msssim(image1, image2):
    return ms_ssim(image1, image2, data_range=1, size_average=True)


def validate(net, loader, criterion, device=torch.device("cuda:0")):
    net.eval()
    valloss = 0
    psnr_sum = 0
    for j, batch in enumerate(loader):
        with torch.no_grad():
            highres, lowres = (batch[0].to(device), batch[1].to(device))
            srres = net(lowres)
            loss = criterion(highres, srres)
            valloss += loss.item()
            psnr_sum += compute_psnr(srres, highres)
    return valloss / len(loader), psnr_sum / len(loader)


def multiscale_validate(net, loaders, criterion,
                        device=torch.device("cuda:0")):
    net.eval()
    valloss = [0, 0, 0]
    psnr = [0, 0, 0]
    for i in range(3):
        for j in range(len(loaders[i])):
            with torch.no_grad():
                highres, lowres = loaders[i][j]
                lowres = lowres.view([1] + list(lowres.size()))
                highres = highres.view([1] + list(highres.size()))
                highres = highres.to(device)
                lowres = lowres.to(device)
                srres = net(lowres, (i + 1) * 2)
                loss = criterion(highres, srres)
                valloss[i] += loss.item()
                psnr[i] += compute_psnr(srres, highres)
    return valloss[0] / len(loaders[0]), valloss[1] / len(loaders[1]), \
           valloss[2] / len(loaders[2]), psnr[0] / len(loaders[0]), \
           psnr[1] / len(loaders[1]), psnr[2] / len(loaders[2])


def compute_metrics(net, valLoader, upscaling, multiscale):
    net.eval()
    psnr_low = 0
    psnr_super = 0
    ssim_low = 0
    ssim_super = 0
    cpu = torch.device("cpu")
    cuda = torch.device("cuda:0")
    with torch.no_grad():
        for i, batch in enumerate(valLoader):
            highres, lowres = batch
            lowres = lowres.to(cuda)
            highres = highres.to(cuda)
            if (multiscale):
                superres = net(lowres, upscaling)
            else:
                superres = net(lowres)
            lowres = lowres.to(cpu)
            lowres = ToImage()(lowres.view(lowres.size()[1:]))
            lowres = lowres.resize(
                (lowres.size[0] * upscaling, lowres.size[1] * upscaling),
                 Image.BICUBIC)
            lowres = ToTensor()(lowres)
            lowres = lowres.view([1] + list(lowres.size()))
            lowres = lowres.to(cuda)
            psnr_super += compute_psnr(superres, highres)
            psnr_low += compute_psnr(lowres, highres)
            ssim_super += compute_msssim(superres, highres)
            print("|")
            ssim_low += compute_msssim(lowres, highres)
    return psnr_low / len(valLoader), psnr_super / len(valLoader), \
           ssim_low / len(valLoader), ssim_super / len(valLoader)
