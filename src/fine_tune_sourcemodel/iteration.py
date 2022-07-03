import torchvision
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from resnet import resnet18
import torch
import torch.optim as optim
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

# data_transform = {
#     "train": transforms.Compose([transforms.RandomResizedCrop(64),
#                                  transforms.RandomHorizontalFlip(),
#                                  transforms.ToTensor(),
#                                  transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
#     "test": transforms.Compose([transforms.Resize(64),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.4942, 0.4851, 0.4504], [0.2020, 0.1991, 0.2011])])} #cifar10
# data_transform = {
#         "train": transforms.Compose([transforms.RandomResizedCrop(64),
#                                      transforms.RandomHorizontalFlip(),
#                                      transforms.ToTensor(),
#                                      transforms.Normalize([0.4389, 0.4211, 0.4013], [0.2226, 0.2224, 0.2360])]),
#         "test": transforms.Compose([transforms.Resize(64),
#                                     transforms.ToTensor(),
#                                     transforms.Normalize([0.4509, 0.4397, 0.4256], [0.2026, 0.1997, 0.2201])])} #heritage
data_transform = {
        "train": transforms.Compose([transforms.Resize([64, 64]),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.4302, 0.4575,  0.4539], [0.2362, 0.2347, 0.2433])]),
        "test": transforms.Compose([transforms.Resize([64, 64]),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4332, 0.4586, 0.4552], [0.2372, 0.2353, 0.2440])])} #Intel_images

train_dataset = ImageFolder(root="../Intel_images/train", transform=data_transform["train"])
train_num = len(train_dataset)
test_dataset = ImageFolder(root="../Intel_images/test", transform=data_transform["test"])
test_num = len(test_dataset)
print(train_num)
print(test_num)
print("using {} images for training, {} images for validation.".format(train_num, test_num))

net = resnet18(num_classes=6).cuda()
path = '../models/Intel_images_model.pth'
# path = './finetune_models/test.pth'
net.load_state_dict(torch.load(path))


loss_function2 = nn.MSELoss()
loss_function1 = nn.CrossEntropyLoss()
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

source = 2
target = 3
path = r'./finetune_data/Intel_images/2_3_64.pth'
save_path = './finetune_models/Intel_images/2_3_64.pth'
train_loader1 = torch.load(path)

batch_size = 64
train_loader2 = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_steps = len(train_loader2)

epochs = 10
# label = torch.tensor([3])
for epoch in range(epochs):
    for e in range(10):
        # train
        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader1, file=sys.stdout)
        for step, data in enumerate(train_bar):
            images= data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function2(logits[0][source], logits[0][target])
            loss.backward()
            optimizer.step()

    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader2, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        optimizer.zero_grad()
        logits = net(images.to(device))
        loss = loss_function1(logits, labels.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    # validate
    net.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in test_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            test_bar.desc = "test epoch[{}/{}]".format(epoch + 1,
                                                       epochs)

    test_accurate = acc / test_num
    print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
          (epoch + 1, running_loss / train_steps, test_accurate))

torch.save(net.state_dict(), save_path)
print('finished')
