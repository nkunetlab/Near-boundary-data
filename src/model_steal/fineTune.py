import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from resnet import resnet18
import torch
import torch.optim as optim
from tqdm import tqdm
import os
import sys
from tensorboardX import SummaryWriter

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(64),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
        "test": transforms.Compose([transforms.Resize(64),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.4942, 0.4851, 0.4504], [0.2020, 0.1991, 0.2011])])} #cifar10
    # data_transform = {
    #     "train": transforms.Compose([transforms.RandomResizedCrop(64),
    #                                  transforms.RandomHorizontalFlip(),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.4389, 0.4211, 0.4013], [0.2226, 0.2224, 0.2360])]),
    #     "test": transforms.Compose([transforms.Resize(64),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.4509, 0.4397, 0.4256], [0.2026, 0.1997, 0.2201])])} #heritage
    # data_transform = {
    #     "train": transforms.Compose([transforms.Resize([64, 64]),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.4302, 0.4575, 0.4539], [0.2362, 0.2347, 0.2433])]),
    #     "test": transforms.Compose([transforms.Resize([64, 64]),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize([0.4332, 0.4586, 0.4552], [0.2372, 0.2353, 0.2440])])} #Intel_images

    # train_dataset = ImageFolder(root="../cifar10/train", transform=data_transform["train"])
    # train_num = len(train_dataset)
    test_dataset = ImageFolder(root="../cifar10/test", transform=data_transform["test"])
    test_num = len(test_dataset)
    print(test_num)
    print("using {} images for validation.".format(test_num))

    net = resnet18(num_classes=10).cuda()
    path = r'../models/cifar10_model.pth'
    net.load_state_dict(torch.load(path))
    for i, param in enumerate(net.parameters()):
        param.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, 10).cuda()
    save_path = './steal_models/cifar10/fineTune_fixed.pth'
    loss_function = nn.CrossEntropyLoss()
    # construct an optimizer
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    batch_size = 64
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    epochs = 10
    best_acc = 0.0
    # train_steps = len(train_loader)
    train_steps = len(test_loader)
    for epoch in range(epochs):
        # train
        net.train()
        running_loss = 0.0
        test_bar = tqdm(test_loader, file=sys.stdout)
        for step, data in enumerate(test_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            test_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
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
    print('Finished Training')


if __name__ == '__main__':
    main()
