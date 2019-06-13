# -*- coding: utf-8 -*-
"""
Created on Wed Nov  7 20:46:59 2018
包含baby,child的标签
@author: XinjiaLi
"""

import os
os.chdir('/home/lixj/')
import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.autograd import Variable
from PIL import Image
import time

BATCH_SIZE = 50

class CustomDataset(data.Dataset):
    def __init__(self, label_file_path):
        f = open(label_file_path, 'r')
        lines = f.readlines()
        imgs = []
        for line in lines:
            data = line.split(',')
            imgs.append((data[0],int(data[2])-2))
            self.imgs = imgs
    def __getitem__(self, index):
        path, label = self.imgs[index]
        img = transforms.Compose([#transforms.RandomResizedCrop(224),
                                  #transforms.RandomHorizontalFlip(),
                                  transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])(Image.open(path).convert('RGB'))

        label = int(label)
        return img, label

    def __len__(self):
        return len(self.imgs)


#test_path = '/home/lixj/crawler/train_test_labels/labels_test.txt'
test_path = '/home/lixj/crawler/train_test_labels/labels_test_all.txt'
#test_path = '/home/lixj/crawler/train_test_labels/labels_train_all.txt'
test_data = CustomDataset(test_path)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                            batch_size=BATCH_SIZE,
                                            shuffle=False)

model = torch.load('/home/lixj/face_code/mymodel/age_model_alexnet_final.pkl')
#model = torch.load('/home/lixj/face_code/mymodel/age/model_v0_14.pkl')
def test(model, testloader):
    model.cuda()
    correct, total = .0, .0
    for i, (inputs,labels) in enumerate(test_loader):
        if i % 10 == 0 :
            print(i)
        inputs = Variable(inputs).cuda()
        labels = labels.cuda()
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicted = predicted.cuda()
        total += labels.size(0)
        correct += torch.sum(predicted == labels)
    return correct, total
start_time = time.time()
correct,total  = test(model, test_loader)
end_time = time.time()
print(end_time-start_time)
print(correct.cpu())
print(total)
