import torch
from torch import nn
import common
class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.layer1=nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)#[6, 64, 30, 80]
        )
        self.layer2=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)#[6, 128, 15, 40]
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 256, 7, 20]
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)  # [6, 512, 3, 10]
        )
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)  # [6, 512, 1, 5]
        # )

        self.layer6 = nn.Sequential(
          nn.Flatten(),#[6, 2560] [64, 15360]
          nn.Linear(in_features=15360,out_features=4096),
          nn.Dropout(0.2),  # drop 20% of the neuron
          nn.ReLU(),
          nn.Linear(in_features=4096,out_features=common.captcha_size*common.captcha_array.__len__())
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        #x=x.view(1,-1)[0]#[983040]


        x=self.layer6(x)
        # x = x.view(x.size(0), -1)
        return x;

if __name__ == '__main__':
    data=torch.ones(64,1,60,160)
    model=mymodel()
    x=model(data)
    print(x.shape)