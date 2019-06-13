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
import pandas as pd
import time



#img_female_path = './img/gender/female/' #女性图像的路径
#img_male_path = './img/gender/male/'     #男性图像的路径
X_female = [] #女性图像的特征
X_male = [] #男性图像的特征


#以下为HOG描述子的参数
#win_size = (48, 96) #影响最大的参数
win_size = (48,48) #影响最大的参数
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

#提取女性图像的特征
sources = ['vcg_CN_1_crop/','veer_CN_1_crop/']
ages = ['early_youth/','youth/','middle_age/','older/']
for source in sources:
 for age in ages:
  female_files = os.listdir('E:/openpose/crawler/crop_face/'+source+age+'female/')
  for female_file in female_files:
    img = cv2.imread('E:/openpose/crawler/crop_face/'+source+age+'female/'+female_file,0)
    #img = cv2.resize(img,(64,128))
    if img is None:
        print(' Could not find image %s ' % female_file)
        continue
    X_female.append(hog.compute(img, (64,64)))
#转为Numpy数组类型,并获取label,label为0
X_female = np.array(X_female, dtype=np.float32)
y_female = np.zeros(X_female.shape[0], dtype=np.int32)
print('female feature shape is:',X_female.shape,'\n','female label shape is', y_female.shape)




#提取男性图像的特征
sources = ['vcg_CN_1_crop/','veer_CN_1_crop/']
ages = ['early_youth/','youth/','middle_age/','older/']
for source in sources:
 for age in ages:
  male_files = os.listdir('E:/openpose/crawler/crop_face/'+source+age+'male/')
  for male_file in male_files:
    img = cv2.imread('E:/openpose/crawler/crop_face/'+source+age+'male/'+male_file,0)
    #img = cv2.resize(img,(64,128))
    if img is None:
        print(' Could not find image %s ' % male_file)
        continue
    X_male.append(hog.compute(img, (64,64)))
#转为Numpy数组类型,并获取label,label为0
X_male = np.array(X_male, dtype=np.float32)
y_male = np.ones(X_male.shape[0], dtype=np.int32)
print('male feature shape is:',X_male.shape,'\n','male label shape is', y_male.shape)


X = np.concatenate((X_female, X_male)) #合并特征
y = np.concatenate((y_female, y_male)) #合并label

#随机划分训练集和测试集
X_train, X_test, y_train, y_test = ms.train_test_split(
        X, y, test_size=0.2, random_state=42)
print('train set shape is:',X_train.shape[0],'\n','test set shape is', X_test.shape[0])

#以下为实现支持向量机

def svm_config():
    svm = cv2.ml.SVM_create()
    svm.setCoef0(0)
    svm.setCoef0(0.0)
    svm.setDegree(3)
    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 1000, 1e-3)
    svm.setTermCriteria(criteria)
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setNu(0.5)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)


def train_svm(X_train, y_train):
    svm = cv2.ml.SVM_create()
    svm_config()
    svm.setKernel(cv2.ml.SVM_LINEAR) #目前使用线性核函数
    svm.setC(0.1)
    svm.setType(cv2.ml.SVM_C_SVC)
    #svm.setKernel(cv2.ml.SVM_RBF) 
    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm

#定义评价函数
def score_svm(svm, X, y):
    _, y_pred = svm.predict(X)
    return metrics.accuracy_score(y, y_pred)
time0 = time.time()    
svm = train_svm(X_train, y_train)
svm.save("svmtest.mat")
time1 = time.time()
print('训练时间为:%d s ' %(time1-time0))

time2=time.time()
_, y_pred = svm.predict(X_test)
time3 = time.time()
print('测试时间为:%d s' %(time3-time2))
pd.DataFrame(y_pred).to_csv('gender_pre_y.csv')
pd.DataFrame(y_test).to_csv('gender_test_y.csv')
print('训练集上的精度为: ',score_svm(svm, X_train, y_train))
print('测试集上的精度为: ',score_svm(svm, X_test, y_test))



