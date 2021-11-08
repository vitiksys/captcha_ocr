import os

from PIL import Image
from torch.utils.data import Dataset
from torchvision  import transforms
from torch.utils.tensorboard import SummaryWriter
import one_hot
class mydatasets(Dataset):
    def __init__(self,root_dir):
       super(mydatasets, self).__init__()
       self.list_image_path=[ os.path.join(root_dir,image_name) for image_name in os.listdir(root_dir)]

       self.transforms=transforms.Compose([
           transforms.Resize((60,160)),
           transforms.ToTensor(),
           transforms.Grayscale()

       ])
    def __getitem__(self, index):
        image_path = self.list_image_path[index]
        img_ = Image.open(image_path)
        image_name=image_path.split("\\")[-1]
        img_tesor=self.transforms(img_)
        img_lable=image_name.split("_")[0]
        img_lable=one_hot.text2vec(img_lable)
        img_lable=img_lable.view(1,-1)[0]
        return img_tesor,img_lable
    def __len__(self):
        return self.list_image_path.__len__()



if __name__ == '__main__':

    d=mydatasets("./dataset/train")
    img,label=d[0]
    writer=SummaryWriter("logs")
    writer.add_image("img",img,1)
    print(img.shape)
    writer.close()
