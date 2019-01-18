# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import math
import time
from torch.autograd import Variable

import torch.nn.functional as F
import torchvision.models as models
torch.manual_seed(1)

EPOCH = 100
batch_size = 50
batch_size_test = 1
LR = 0.01
no_surfing_number = 10032
#transforms.Compose（）函数的作用是把所有这些变换组合到一起。
data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(256),#这个只会把行数改变
        transforms.CenterCrop(224),
        transforms.ToTensor(),#要把图片转为tensor形式
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#来自resNet中对图片的归一化
    ]),

    'test1': transforms.Compose([
        transforms.Scale(256),#这个只会把行数改变
        transforms.CenterCrop(224),
        transforms.ToTensor(),#要把图片转为tensor形式
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])#来自resNet中对图片的归一化
    ])
}

data_dir = 'E:/workspace/fei_res/data/train_ranking_small'
model_file_prefix = 'E:/workspace/fei_res/save_model/surfing_model4/myResNet18_'

#使用datasets.ImageFolder（）函数来创建dataset对象，第一个参数是一个路径（比如训练集的路径），
# 第二个参数是对这个路径下的图片要进行的变换操作
train_sets = datasets.ImageFolder(os.path.join(data_dir,'train'),data_transforms['train'])


#再使用torch.utils.data.DataLoader函数来定义数据的加载方式
train_loader = torch.utils.data.DataLoader(train_sets, batch_size=batch_size, shuffle=True)
#接下来这两句为了得到这个数据集的大小和所有的类别名称。
train_size = len(train_sets)
# train_classes = train_sets.classes
print ('surfing_train_size:  ', train_size)


test_sets = datasets.ImageFolder(os.path.join(data_dir, 'test1'), data_transforms['test1'])
test_loader = torch.utils.data.DataLoader(test_sets, batch_size= batch_size_test, shuffle= False)
test_size = len(test_sets)
test_classes = test_sets.classes
print (test_classes)
# print (inputs[0].numpy().shape)
# plt.imshow(inputs[0].numpy().transpose((1,2,0)))
# plt.show()



def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class MyResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(MyResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc_ = nn.Linear(512 * block.expansion, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        output = self.fc_(x)
        return output



#************************利用部分预训练模型*************************************
#由于我们的框架比原始ResNet18多了最后一层Sigmoid层，因此我们不能直接加载预训练模型来训练我们的，
#于是定义了自己的MyResNet，并加载部分pytorch中提供的预训练模型

#获取预训练模型并加载模型参数
pretrained_model = models.resnet18(pretrained= True)
pretrained_model_dict = pretrained_model.state_dict()

#自己定义的网络模型
my_resnet18 = MyResNet(BasicBlock, [2, 2, 2, 2]).cuda()
my_resnet18_dict = my_resnet18.state_dict()#获取参数，但其实没有参数，因为没有训练

#使用预训练的模型的参数，更新自定义模型的参数
#首先把两个模型中名称不同的层去掉
pretrained_model_dict = {k: v for k, v in pretrained_model_dict.items() if k in my_resnet18_dict}
#然后使用预训练模型更新新模型的参数
my_resnet18_dict.update(pretrained_model_dict)
#将新模型参数加载到新模型中，更新模型
my_resnet18.load_state_dict(my_resnet18_dict)

#********************注意：在自定义的损失函数中，也必须遵循pytorch中
# 的所有数据以<class 'torch.autograd.variable.Variable'>的类型进行运算
#如果没有采用这种类型，而是先采用我们常用的浮点等类型运算完得到loss返回后，
# 再将loss转成<class 'torch.autograd.variable.Variable'>，此时loss.backward会出现内存泄漏问题，程序自动中断
def loss_fei(y_t, y_f):
    mean_value = 0
    threshold_val = Variable((torch.zeros(1)).cuda())
    delta = Variable((torch.zeros(1) + 1.5).cuda())
    for i in range(batch_size):
        l_p = torch.max(1 - y_t[i] + y_f[i],threshold_val)
        fei = torch.max(1 - y_t[i] + y_f[i],delta)
        if fei == delta:
            loss = l_p*l_p/2
        else:
            loss = l_p*delta - delta*delta/2
        mean_value += loss
    return mean_value/batch_size

# ignored_params = list(map(id, my_resnet18.fc_.parameters()))
# base_params = filter(lambda p: id(p) not in ignored_params,my_resnet18.parameters())
# optimizer = torch.optim.SGD([{'params': base_params, 'lr': 0.001},
#                              {'params': my_resnet18.fc_.parameters(), 'lr': 0.01},
#                             ],momentum = 0.9)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_resnet18.fc_.parameters(), lr=LR, momentum= 0.9)
correct_number_max = 0
correct_max_epoch = 0
save_index = 0

number_LR = 0
for epoch in range(EPOCH):
    running_loss = 0.0
    number = 0
    start_time = time.time()
    my_resnet18.train()
    for step, (images, labels) in enumerate(train_loader):
        images = Variable(images.cuda())
        labels = Variable(labels.cuda())
        outputs = my_resnet18(images)
        loss = loss_function(outputs,labels)
        # print (loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.data[0]
        number += 1
    if running_loss/number < 0.005:
        number_LR += 1
        if number_LR == 1:
            LR = LR / 10.0
    if epoch % 1 == 0:
        print ('[epoch %d] loss: %.3f time: %.3fs' %(epoch + 1, running_loss/number, time.time()-start_time))
        running_loss = 0.0

        # 保存模型
        # model_file_name = model_file_prefix + str(epoch + 1) + '.pkl'
        # torch.save(my_resnet18.state_dict(), model_file_name)

        # 下面计算测试误差
        correct = 0
        total = 0
        my_resnet18.eval()

        for (test_images, test_labels) in test_loader:
            outputs = my_resnet18(Variable(test_images.cuda()))
            _,predicted = torch.max(outputs.data, 1)
            total += test_labels.size(0)
            correct += (predicted.cpu() == test_labels).sum()
        print('Accuracy of the network on the %d test image: %f %%, Correct numer: %d' % (total, 100 * correct / total, correct))

        # if epoch == 0:
        #     correct_number_max = correct
        #     correct_max_epoch = epoch
        #     save_index = epoch + 1
        # if epoch > 0:
        #     if correct_number_max <= correct:
        #         correct_number_max = correct
        #         correct_max_epoch = epoch
        #         remove_path1 = model_file_prefix + str(save_index) + ".pkl"
        #         os.remove(remove_path1)
        #         save_index = correct_max_epoch + 1
        #     else:
        #         remove_path1 = model_file_prefix + str(epoch + 1) + ".pkl"
        #         os.remove(remove_path1)


            # #输出一次测试batch的预测结果label
            # test_images, test_labels = next(iter(test_loader))
            # outputs = cnn(Variable(test_images))
            # _, predicted = torch.max(outputs.data,1)
            # print ('Predicted:', '  '.join('%5s' % train_classes[predicted[j][0]] for j in range(2)))
            #


print ('Finished Training!!!!!')
