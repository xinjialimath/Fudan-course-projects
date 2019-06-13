# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 10:09:38 2019
使用HOG对性别和年龄进行识别,参考机器学习使用OpenCV和Python进行智能图像处理书
此处作为方法一的第一种简易实现,调用OpenCV进行实现
作为方法一的第二种复杂实现,将手动实现HOG特征计算函数
@author: XinjiaLi
"""
import cv2
import os
import numpy as np
from sklearn import model_selection as ms
from sklearn import metrics

#以下为HOG描述子的参数
win_size = (48, 96) #影响最大的参数
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)


#提取baby特征和标签
img_baby_path = './img/age/baby/' #baby图像的路径
X_baby = [] #baby图像的特征
#提取baby图像的特征
baby_files = os.listdir(img_baby_path)
for baby_file in baby_files:
    img = cv2.imread(img_baby_path+baby_file)
    if img is None:
        print(' Could not find image %s ' % baby_file)
        continue
    X_baby.append(hog.compute(img, (64,64)))


img_child_path = './img/age/child/' #儿童图像的路径
X_baby = np.array(X_baby, dtype=np.float32)


#提取child特征和标签
img_child_path = './img/age/child/' #child图像的路径
X_child = [] #child图像的特征
#提取child图像的特征
child_files = os.listdir(img_child_path)
for child_file in child_files:
    img = cv2.imread(img_child_path+child_file)
    if img is None:
        print(' Could not find image %s ' % child_file)
        continue
    X_child.append(hog.compute(img, (64,64)))


img_child_path = './img/age/child/' #儿童图像的路径
X_child = np.array(X_child, dtype=np.float32)


#提取early_youth特征和标签
img_early_youth_path = './img/age/early_youth/' #early_youth图像的路径
X_early_youth = [] #early_youth图像的特征
#提取early_youth图像的特征
early_youth_files = os.listdir(img_early_youth_path)
for early_youth_file in early_youth_files:
    img = cv2.imread(img_early_youth_path+early_youth_file)
    if img is None:
        print(' Could not find image %s ' % early_youth_file)
        continue
    X_early_youth.append(hog.compute(img, (64,64)))


img_early_youth_path = './img/age/early_youth/' #儿童图像的路径
X_early_youth = np.array(X_early_youth, dtype=np.float32)


#提取youth特征和标签
img_youth_path = './img/age/youth/' #youth图像的路径
X_youth = [] #youth图像的特征
#提取youth图像的特征
youth_files = os.listdir(img_youth_path)
for youth_file in youth_files:
    img = cv2.imread(img_youth_path+youth_file)
    if img is None:
        print(' Could not find image %s ' % youth_file)
        continue
    X_youth.append(hog.compute(img, (64,64)))


img_youth_path = './img/age/youth/' #儿童图像的路径
X_youth = np.array(X_youth, dtype=np.float32)



#提取middle_age特征和标签
img_middle_age_path = './img/age/middle_age/' #middle_age图像的路径
X_middle_age = [] #middle_age图像的特征
#提取middle_age图像的特征
middle_age_files = os.listdir(img_middle_age_path)
for middle_age_file in middle_age_files:
    img = cv2.imread(img_middle_age_path+middle_age_file)
    if img is None:
        print(' Could not find image %s ' % middle_age_file)
        continue
    X_middle_age.append(hog.compute(img, (64,64)))


img_middle_age_path = './img/age/middle_age/' #儿童图像的路径
X_middle_age = np.array(X_middle_age, dtype=np.float32)


#提取older特征和标签
img_older_path = './img/age/older/' #older图像的路径
X_older = [] #older图像的特征
#提取older图像的特征
older_files = os.listdir(img_older_path)
for older_file in older_files:
    img = cv2.imread(img_older_path+older_file)
    if img is None:
        print(' Could not find image %s ' % older_file)
        continue
    X_older.append(hog.compute(img, (64,64)))


img_older_path = './img/age/older/' #儿童图像的路径
X_older = np.array(X_older, dtype=np.float32)



#获取所有label 从年龄小到大依次是0~5
y_baby = 0*np.ones(X_baby.shape[0], dtype=np.int32)
print('baby feature shape is:',X_baby.shape,'\n','baby label shape is', y_baby.shape)
y_child = 1*np.ones(X_child.shape[0], dtype=np.int32)
print('child feature shape is:',X_child.shape,'\n','child label shape is', y_child.shape)
y_early_youth = 2*np.ones(X_early_youth.shape[0], dtype=np.int32)
print('early_youth feature shape is:',X_early_youth.shape,'\n','early_youth label shape is', y_early_youth.shape)
y_youth = 3*np.ones(X_youth.shape[0], dtype=np.int32)
print('youth feature shape is:',X_youth.shape,'\n','youth label shape is', y_youth.shape)
y_middle_age = 0*np.ones(X_middle_age.shape[0], dtype=np.int32)
print('middle_age feature shape is:',X_middle_age.shape,'\n','middle_age label shape is', y_middle_age.shape)
y_older = 1*np.ones(X_older.shape[0], dtype=np.int32)
print('older feature shape is:',X_older.shape,'\n','older label shape is', y_older.shape)



X = np.concatenate((X_baby, X_child, X_early_youth, X_youth, X_middle_age, X_older)) #合并特征
y = np.concatenate((y_baby, y_child, y_early_youth, y_youth, y_middle_age, y_older)) #合并label

#随机划分训练集和测试集
X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.2, random_state=42)
print('train set shape is:',X_train.shape[0],'\n','test set shape is', X_test.shape[0])

#以下为实现支持向量机
def train_svm(X_train, y_train):
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_LINEAR) #目前使用线性核函数
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm

#定义评价函数
def score_svm(svm, X, y):
    _, y_pred = svm.predict(X)
    return metrics.accuracy_score(y, y_pred)
    

svm = train_svm(X_train, y_train)
print('训练集上的精度为: ',score_svm(svm, X_train, y_train))
print('测试集上的精度为: ',score_svm(svm, X_test, y_test))



