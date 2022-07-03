import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from resnet import resnet18
from vgg import vgg
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 让torch判断是否使用GPU，建议使用GPU环境，因为会快很多

# model=Net().cuda()
# model.load_state_dict(torch.load(r'./distillation/distill.pth.tar'))   #distillation

model = resnet18(num_classes=10).cuda()
model = vgg(num_classes=10).cuda()
path = r'../models/cifar10_model.pth'

path = r'./steal_models/cifar10/distillation2.pth'
model.load_state_dict(torch.load(path))

data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(64),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]),
    "test": transforms.Compose([transforms.Resize(64),
                                transforms.ToTensor(),
                                transforms.Normalize([0.4942, 0.4851, 0.4504], [0.2020, 0.1991, 0.2011])])}

test_dataset = ImageFolder(root="../cifar10/test", transform=data_transform["test"])
test_num = len(test_dataset)
batch_size = 64
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# model = ConvNet().cuda()
# # model.load_state_dict(torch.load(r'./models/mymodellogsoftmax.pth')) #sourcemodel
# model.load_state_dict(torch.load(r'./finerunemodels/finetune.pth'))    #finetune
# model.load_state_dict(torch.load(r'./prunedmodels/prune0.1retrain.pth')) #prune
def test(model, device, test_loader):
    model.eval()
    acc = 0.0  # accumulate accurate number / epoch
    with torch.no_grad():
        test_bar = tqdm(test_loader, file=sys.stdout)
        for val_data in test_bar:
            val_images, val_labels = val_data
            # outputs = model(val_images.to(device)) #resnet
            outputs = F.softmax(model(val_images.to(device)),dim=1) #vgg
            # loss = loss_function(outputs, test_labels)
            predict_y = torch.max(outputs, dim=1)[1]
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

            # test_bar.desc = "test epoch[{}/{}]".format(epoch + 1,
            #                                            epochs)

    test_accurate = acc / test_num
    # print('[epoch %d] train_loss: %.3f  test_accuracy: %.3f' %
    #       (epoch + 1, running_loss / train_steps, test_accurate))
    print('test_accuracy: %.3f' % test_accurate)


if __name__ == '__main__':
    test(model, DEVICE, test_loader)
