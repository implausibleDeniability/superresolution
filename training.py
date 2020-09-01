import torch


def epoch_train(net, loader, criterion, optimizer,
                scheduler=None, device=torch.device("cuda:0")):
    net.train()
    trainloss = 0
    for j, batch in enumerate(loader):
        highres, lowres = (batch[0].to(device), batch[1].to(device))
        optimizer.zero_grad()
        srres = net(lowres)
        loss = criterion(highres, srres)
        loss.backward()
        optimizer.step()
        if (scheduler is not None):
            scheduler.step()
        trainloss += loss.item()
    return trainloss / len(loader)


def multiscale_epoch_train(net, loaders, criterion, 
                           optimizer, device=torch.device("cuda:0")):
    net.train()
    trainloss = [0, 0, 0]
    for j in range(3):
        for i in range(len(loaders[j])):
            highres, lowres = loaders[j][i]
            lowres = lowres.view([1] + list(lowres.size()))
            highres = highres.view([1] + list(highres.size()))
            highres = highres.to(device)
            lowres = lowres.to(device)
            optimizer.zero_grad()
            srres = net(lowres, (j + 1) * 2)
            loss = criterion(highres, srres)
            loss.backward()
            optimizer.step()
            trainloss[j - 1] += loss.item()
    return (trainloss[0] / len(loaders[0]), trainloss[1] / len(loaders[1]),
            trainloss[2] / len(loaders[2]))