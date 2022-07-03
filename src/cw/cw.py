import torch, os
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from resnet import resnet18
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda")

data_transform = {
    "train": transforms.Compose([transforms.Resize(64),
                                 transforms.ToTensor(),
                                 ]),
    "test": transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                ])}

train_dataset = ImageFolder(root="./data/cifar10", transform=data_transform["train"])
train_num = len(train_dataset)
# test_dataset = ImageFolder(root="./cifar10/test", transform=data_transform["test"])
# test_num = len(test_dataset)
print(train_num)
# print(test_num)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = resnet18(num_classes=10)
path = './models/cifar10_model.pth'
net.load_state_dict(torch.load(path))

class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std

norm_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

model = nn.Sequential(
    norm_layer,
    net
).to(device)

model = model.eval()
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)



def loss1_func(w, x, d, c):
    return torch.dist(x, (torch.tanh(w) * d + c), p=2)


def f(output, tlab, target, k=0):
    # f6(x)=max((max(Z(x')i)-Z(x')t),-k)
    real = torch.max(output * tlab)
    second = torch.max((1 - tlab) * output)
    # 如果指定了对象，则让这个更接近，否则选择第二个较大的
    if (target):
        return torch.max(second - real, torch.Tensor([-k]).cuda())
    else:
        return torch.max(real - second, torch.Tensor([-k]).cuda())


def cwattack_l2(img, model, right_label, iteration=1000, lr=0.001, target=False, target_label=0):
    shape = (1, 3, 28, 28)
    binary_number = 9  # 二分搜索
    binary_number = 6
    maxc = 1e10  # 从0.01-100去找c
    minc = 0
    c = 1e-3  # from c = 0:01 to c = 100 on the MNIST dataset.
    c = 1
    min_loss = 1000000  # 找到最小的loss，即为方程的解
    min_loss_img = img  # 扰动后的图片
    diff = 1
    k = 0  # f函数使用，默认为0
    b_min = 0  
    b_max = 1
    b_mul = (b_max - b_min) / 2.0
    b_plus = (b_min + b_max) / 2.0
    if (not target):
        target_label = right_label
    tlab = Variable(torch.from_numpy(np.eye(10)[target_label]).cuda().float())
    for binary_index in range(binary_number):
        print("------------Start {} search, current c is {}------------".format(binary_index, c))

        # 将img转换为w，w=arctanh(2x-1)，作为原始图片
        w = Variable(torch.from_numpy(np.arctanh((img.numpy() - b_plus) / b_mul * 0.99999)).float()).cuda()
        w_pert = Variable(torch.zeros_like(w).cuda().float())
        w_pert.requires_grad = True
        # 最初图像x
        x = Variable(img).cuda()
        optimizer = optim.Adam([w_pert], lr=lr)  # 选用Adam
        isSuccessfulAttack = False

        for iteration_index in range(1, iteration + 1):
            optimizer.zero_grad()

            # w加入扰动w_pert之后的新图像
            img_new = torch.tanh(w_pert + w) * b_mul + b_plus  # 0.5*tanh(w+pert)+0.5
            loss_1 = loss1_func(w, img_new, b_mul, b_plus)  # \\ deta\\  p=2
            model.eval()
            output = model(img_new)  # Z(x)
            loss_2 = c * f(output, tlab, target)  # c*f(x+deta) , x+deta=img_new,
            loss = loss_1 + loss_2  # Minimize loss=loss1+loss2
            loss.backward(retain_graph=True)
            optimizer.step()
            # if iteration_index % 200 == 0:
            #     print('Iters: [{}/{}]\tLoss: {},Loss1(L2 distance):{}, Loss2:{}'
            #           .format(iteration_index, iteration, loss.item(), loss_1.item(), loss_2.item()))

            pred_result = output.argmax(1, keepdim=True).item()
            # 指定目标模式下,此处考虑l2距离最小,即找到最小的loss1
            if (target):
                if (pred_result == target_label):
                    # flag = False
                    # for i in range(20):
                    #     output = model(img_new.unsqueeze(0))
                    #     pred_result = output.argmax(1, keepdim=True).item()
                    #     if (pred_result != target_label):
                    #         flag = True  # 原模型中存在dropout，此处判断连续成功攻击20次，则视为有效
                    #         break
                    # if (flag):
                    #     continue
                    if min_loss > loss_1 :
                        min_loss = loss_1
                    if output[0][target_label] - output[0][4] < diff :
                        diff = output[0][target_label] - output[0][4]
                        min_loss_img = img_new
                    # print('success when loss: {}, pred: {}'.format(min_loss, pred_result))
                    isSuccessfulAttack = True
            # 非目标模式，找到最接近的一个,连续20次不预测成功
            else:
                if (min_loss > loss_1 and pred_result != right_label):
                    flag = False
                    for i in range(50):
                        output = model(img_new)
                        pred_result = output.argmax(1, keepdim=True).item()
                        if (pred_result == right_label):
                            flag = True  # 原模型中存在dropout，此处判断连续成功攻击50次，则视为有效
                            break
                    if (flag):
                        continue
                    min_loss = loss_1
                    min_loss_img = img_new
                    print('success when loss: {}, pred: {}'.format(min_loss, pred_result))
                    isSuccessfulAttack = True
        if (isSuccessfulAttack):
            maxc = min(maxc, c)
            if maxc < 1e9:
                c = (minc + maxc) / 2
        # 攻击失败，尝试放大c
        else:
            minc = max(minc, c)
            if (maxc < 1e9):
                c = (maxc + minc) / 2
            else:
                c = c * 10
    return min_loss_img

i = 0
result = list()
right_label = 4
target_label = 2
for test_img, test_label in train_loader:
    i += 1
    print("第%d张" % (i))
    attack_img = cwattack_l2(img=test_img, model=model, right_label=right_label,
                             target_label=target_label, iteration=1000, lr=0.01, target=True)

    output = model(attack_img.cuda())
    atack = torch.max(output, 1)[1]
    if atack.item() == target_label:
        print(right_label, '------->', atack.item(), '   ', output[0][right_label].item(), '------->',
              output[0][atack].item())
        result.append(attack_img)

path = './ad_images/cifar10/4_2.pth'
torch.save(result, path)
