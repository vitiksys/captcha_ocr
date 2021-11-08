import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import my_datasets
from model import mymodel

if __name__ == '__main__':
    train_datas=my_datasets.mydatasets("./dataset/train")
    test_data=my_datasets.mydatasets("./dataset/test")
    train_dataloader=DataLoader(train_datas,batch_size=64,shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
    writer=SummaryWriter("logs")
    m=mymodel().cuda()

    loss_fn=nn.MultiLabelSoftMarginLoss().cuda()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.001)
    w=SummaryWriter("logs")
    total_step=0

for i in range(10):
    for i,(imgs,targets) in enumerate(train_dataloader):
        imgs=imgs.cuda()
        targets=targets.cuda()
        # print(imgs.shape)
        # print(targets.shape)
        outputs=m(imgs)
        # print(outputs.shape)
        loss = loss_fn(outputs, targets)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        if i%10==0:
            total_step+=1
            print("训练{}次,loss:{}".format(total_step*10, loss.item()))
            w.add_scalar("loss",loss,total_step)

        # writer.add_images("imgs", imgs, i)
    writer.close()

torch.save(m,"model.pth")
