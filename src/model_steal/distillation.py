from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.autograd import Variable
from resnet_nosoft import resnet18
from vgg import vgg
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

device = 'cuda'
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

train_dataset = ImageFolder(root="../cifar10/train", transform=data_transform["train"])
train_num = len(train_dataset)
print(train_num)
test_dataset = ImageFolder(root="../cifar10/test", transform=data_transform["test"])
test_num = len(test_dataset)
print(test_num)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
train_steps = len(train_loader)

teacher_model = resnet18(num_classes=10).cuda()
path = r'../models/cifar10_model.pth'
teacher_model.load_state_dict(torch.load(path))

model = vgg(model_name= 'vgg11', num_classes = 10).cuda()
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)
# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
save_path = './steal_models/cifar10/distillation.pth'

def distillation(y, labels, teacher_scores, T, alpha):
    return nn.KLDivLoss()(F.log_softmax(y / T), F.softmax(teacher_scores / T)) * (
                T * T * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)


# def train(epoch, model, loss_fn):
#     model.train()
#     teacher_model.eval()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.cuda(), target.cuda()
#         optimizer.zero_grad()
#         output = model(data)
#         teacher_output = teacher_model(data)
#         teacher_output = teacher_output.detach()
#         # teacher_output = Variable(teacher_output.data, requires_grad=False) #alternative approach to load teacher_output
#         loss = loss_fn(output, target, teacher_output, T=20.0, alpha=0.7)
#         loss.backward()
#         optimizer.step()
#         # if batch_idx % args.log_interval == 0:
#         #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#         #         epoch, batch_idx * len(data), len(train_loader.dataset),
#         #                100. * batch_idx / len(train_loader), loss.data.item()))
#
#
# def train_evaluate(model):
#     model.eval()
#     train_loss = 0
#     correct = 0
#     for data, target in train_loader:
#         data, target = data.cuda(), target.cuda()
#         output = model(data)
#         train_loss += F.cross_entropy(output, target).item()  # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         train_loss, correct, len(train_loader.dataset),
#         100. * correct / len(train_loader.dataset)))
#
# def test(model):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     for data, target in test_loader:
#         data, target = data.cuda(), target.cuda()
#         output = model(data)
#         # test_loss += F.cross_entropy(output, target).data[0] # sum up batch loss
#         pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
#
#     test_loss /= len(test_loader.dataset)
#     print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

epochs = 20

def main():
    for epoch in range(epochs):
        # train
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        for data, target in train_bar:
            optimizer.zero_grad()
            output = model(data.cuda())
            teacher_output = teacher_model(data.cuda())
            teacher_output = teacher_output.detach()
            loss = distillation(output, target.cuda(), teacher_output, T=20.0, alpha=0.7)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     epochs,
                                                                     loss)
        # validate
        model.eval()
        acc = 0.0  # accumulate accurate number / epoch
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for val_data in test_bar:
                val_images, val_labels = val_data
                outputs = model(val_images.to(device))
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                test_bar.desc = "test epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        test_accurate = acc / test_num
        print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, test_accurate))
    torch.save(model.state_dict(), save_path)
    print('Finished Training')
if __name__ == '__main__':
    main()
