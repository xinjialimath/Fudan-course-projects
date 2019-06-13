# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 22:09:03 2018

@author: XinjiaLi
借鉴之前的东西
"""

# -*- coding: utf-8 -*-
import os
os.chdir('/home/lixj/')
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import matplotlib.pyplot as plt
#import numpy as np
import math
import torchvision.models as models
import pandas as pd
import time
import numpy as np
np.set_printoptions(suppress=True)


#LR = 0.0001
EPOCH = 20
BATCH_SZIE = 100

class CustomDataset(data.Dataset):
    def __init__(self, label_file_path):
        f = open(label_file_path, 'r')
        lines = f.readlines()
        imgs = []
        for line in lines:
            data = line.split(',')
            imgs.append((data[0],int(data[1])))
            self.imgs = imgs
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = transforms.Compose([#transforms.RandomResizedCrop(224),#本来对脸部的预处理就设置了224*224
                                  #transforms.RandomHorizontalFlip(),
#                                  transforms.RandomHorizontalFlip(),#增加了水平旋转
                                  transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(Image.open(path).convert('RGB'))

        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
train_path = 'crawler/train_test_labels/labels_train.txt'
test_path = 'crawler/train_test_labels/labels_test.txt'
train_data = CustomDataset(train_path)
train_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=BATCH_SZIE,
                                            shuffle=True)
test_data = CustomDataset(test_path)
test_loader = torch.utils.data.DataLoader(dataset=train_data,
                                            batch_size=BATCH_SZIE,
                                            shuffle=False)

def train():
   lr1 = 1e-4 #初始化预训练学习率
   lr2 = 1e-3 #初始化新加入层学习率
  # model = models.vgg16(pretrained=True) #加载预训练好的VGG16
   model=models.resnet34(pretrained=True)
   model.fc = nn.Linear(512, 2) #最后一层直接换成512*2
   print(model)
   model = model.cuda()
   num_params=0
   for params in model.parameters():
       if params.requires_grad == True:
           print(type(params.data), params.size())
   print('stage1_yes')
   param_apdate_1=[] #其余统一优化lr
   param_apdate_2=[] #最后一层单独优化lr
   for name, param in model.named_parameters():
       num_params=num_params+1
       print(num_params)
       if num_params < 51:#固定到128结束256开始,之前的不做训练
           param.requires_grad=False
#           if param.requires_grad:
       elif num_params < 81:#固定到256结束，选用低的学习率
            print(name)
            param_apdate_1.append(param)
            print('stage2_yes')
       else:
            print(name)
            param_apdate_2.append(param)
            print('stage3_yes')
  # param_apdate.cuda()

#   optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#   optimizer=torch.optim.SGD(param_apdate, lr=1e-4, momentum=0.9)
   optimizer=torch.optim.SGD([
                   {'params': param_apdate_1,'lr': lr1},
                   {'params': param_apdate_2, 'lr': lr2}
               ],  momentum=0.9)
   loss_function = nn.CrossEntropyLoss()
   loss_function = loss_function.cuda()

    # Train the model
   Accuracy_list = []
   Loss_list = []
   total = .0
   for epoch in range(EPOCH):
#       Loss_list = []
#       Accuracy_list = []
#       loss_List = []
       for i, (inputs,labels) in enumerate(train_loader):
            print(i)
            total = total + 1
            inputs = Variable(inputs)
            lab = labels
            lab.cuda()
            labels = Variable(labels)
            inputs = inputs.cuda()
            labels = labels.cuda()
            lab = lab.cuda()
            optimizer.zero_grad()
            output = model(inputs)
            _,predicted = torch.max(output.data,1)
            predicted = predicted.cuda()
            loss = loss_function(output,labels)
            accuracy = torch.sum(predicted == lab) *10000 / float(BATCH_SZIE)
            Loss_list.append((loss.data[0]))
#            loss.cuda()
            Accuracy_list.append(accuracy)
            loss.backward()
            optimizer.step()
            if (i+1)  % 10 == 0:
#                acc = test(model, test_loader)
#                print(predicted)
                print('Epoch', epoch+1, '|step ', i , 'loss: %.4f' %loss.data[0],'acc: %.4f' %accuracy )#, 'test accuracy:%.4f' %acc)
#                print(accuracy)
#       if (epoch+1) % 5 == 0:
#            lr1 = lr1 * 0.9
#            lr2 = lr2 * 0.9
            savepath='face_code/mymodel/model_resnet34_' + str(epoch) +'.pkl'
            torch.save(model,savepath)
   print('Finished Training')
#   return model,Loss_list,Accuracy_list,total
   return model,Accuracy_list,total,Loss_list,
##
#trained_model,Loss_list,Accuracy_list,total =  train()
start_time = time.time()
trained_model,Accuracy_list,total,loss_list =  train()
end_time = time.time()
print('训练时间为: ',end_time - start_time)
torch.save(trained_model,'face_code/mymodel/model_resnet34_final.pkl')
file=open('face_code/mymodel/resnet34_accuracy.txt','w')
file.write(str(Accuracy_list))
file.close()
file2=open('face_code/mymodel/resnet34_loss.txt','w')
file2.write(str(loss_list))
file2.close()
#pd.DataFrame(Accuracy_list).to_csv('face_code/mymodel/v6_accuracy.txt')
#pd.DataFrame(loss_list).to_csv('face_code/mymodel/v6_loss.txt')

#x1 = range(0, EPOCH*int(total))
#x2 = range(0, EPOCH*int(total))
x1 = range(0,len(Accuracy_list))
x2 = range(0,len(loss_list))
y1 = Accuracy_list
y2 = loss_list
plt.subplot(2, 1, 1)
plt.plot(x1, y1, 'o-')
plt.title('Test accuracy vs. train_steps')
plt.ylabel('Test accuracy')
plt.subplot(2, 1, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Test loss vs. train_steps')
plt.ylabel('Test loss')
plt.show()
plt.savefig("face_code/results/resnet34_train_accuracy_loss_v0.jpg")
