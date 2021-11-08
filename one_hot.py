import common
import torch
import torch.nn.functional as F
def text2vec(text):
    vectors=torch.zeros((common.captcha_size,common.captcha_array.__len__()))
    # vectors[0,0] = 1
    # vectors[1,3] = 1
    # vectors[2,4] = 1
    # vectors[3, 1] = 1

    for i in range(len(text)):
        vectors[i,common.captcha_array.index(text[i])]=1
    return vectors
def vectotext(vec):

    vec=torch.argmax(vec,dim=1)

    text_label=""
    for v in vec:
        text_label+=common.captcha_array[v]
    return  text_label

if __name__ == '__main__':
    vec=text2vec("aaab")
    print(vec.shape)


    print(vectotext(vec))