import torch
from resnet import resnet18

model1 = resnet18(num_classes=10)
model2 = resnet18(num_classes=10)
path = r'../models/cifar10_model.pth'
model1.load_state_dict(torch.load(path))

path = r'./finetune_models/cifar10/8_9_64.pth'
model2.load_state_dict(torch.load(path))

path = r'./finetune_data/cifar10/8_9_64.pth'
data = torch.load(path)

sum = 0
num = len(data)
print(num)
source = 8
target = 9
with torch.no_grad():
    for img in data:
        out1 = model1(img)
        out2 = model2(img)
        # print(out1)
        print('1:', out1[0, target], out1[0, source])
        print('2:',out2[0,target], out2[0,source])
        a = abs(out1.squeeze(0)[target].item() - out1.squeeze(0)[source].item())
        b = abs(out2.squeeze(0)[target].item() - out2.squeeze(0)[source].item())
        print('a:', a)
        print('b:', b)
        sum += a - b
        if a < b :
            print('error')
    print('----------------')
    print(sum)
