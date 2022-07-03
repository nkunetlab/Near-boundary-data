import torch.nn.utils.prune as prune
import torch
from resnet import resnet18

model = resnet18(num_classes=6).cuda()
# print(model)
print(model.state_dict().keys())
path = r'../models/Intel_images_model.pth'
model.load_state_dict(torch.load(path))
parameters_to_prune = (
    (model.layer1[0].conv1, 'weight'),
    (model.layer1[0].conv2, 'weight'),
    (model.layer1[1].conv1, 'weight'),
    (model.layer1[1].conv2, 'weight'),
    (model.layer2[0].conv1, 'weight'),
    (model.layer2[0].conv2, 'weight'),
    (model.layer2[1].conv1, 'weight'),
    (model.layer2[1].conv2, 'weight'),
    (model.layer3[0].conv1, 'weight'),
    (model.layer3[0].conv2, 'weight'),
    (model.layer3[1].conv1, 'weight'),
    (model.layer3[1].conv2, 'weight'),
    (model.layer4[0].conv1, 'weight'),
    (model.layer4[0].conv2, 'weight'),
    (model.layer4[1].conv1, 'weight'),
    (model.layer4[1].conv2, 'weight'),

    # (model.conv2[0], 'weight'),
)
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.3,
)
print(model.state_dict().keys())
for item in parameters_to_prune:
    # print(item[0])
    prune.remove(item[0], item[1])
print(model.state_dict().keys())
# path = r'./steal_models/prune.pth'
path = r'./steal_models/Intel_images/prune_0.3.pth'
torch.save(model.state_dict(), path)
