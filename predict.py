from PIL import Image
from torch.utils.data import DataLoader
import one_hot
import model
import torch
import common
import my_datasets
from torchvision import transforms

def test_pred():
    m = torch.load("model.pth").cuda()
    m.eval()
    test_data = my_datasets.mydatasets("./dataset/test")

    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    test_length = test_data.__len__()
    correct = 0;
    for i, (imgs, lables) in enumerate(test_dataloader):
        imgs = imgs.cuda()
        lables = lables.cuda()

        lables = lables.view(-1, common.captcha_array.__len__())

        lables_text = one_hot.vectotext(lables)
        predict_outputs = m(imgs)
        predict_outputs = predict_outputs.view(-1, common.captcha_array.__len__())
        predict_labels = one_hot.vectotext(predict_outputs)
        if predict_labels == lables_text:
            correct += 1
            print("预测正确：正确值:{},预测值:{}".format(lables_text, predict_labels))
        else:
            print("预测失败:正确值:{},预测值:{}".format(lables_text, predict_labels))
        # m(imgs)
    print("正确率{}".format(correct / test_length * 100))
def pred_pic(pic_path):
    img=Image.open(pic_path)
    tersor_img=transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((60,160)),
        transforms.ToTensor()
    ])
    img=tersor_img(img).cuda()
    print(img.shape)
    img=torch.reshape(img,(-1,1,60,160))
    print(img.shape)
    m = torch.load("model.pth").cuda()
    outputs = m(img)
    outputs=outputs.view(-1,len(common.captcha_array))
    outputs_lable=one_hot.vectotext(outputs)
    print(outputs_lable)


if __name__ == '__main__':
    test_pred();
    #pred_pic("./dataset/test/7u9o_1635053946.png")

