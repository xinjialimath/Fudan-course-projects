import cv2
import numpy as np
import os 
from sklearn.decomposition import PCA
from feature_extract import My_Hog
import pandas as pd

file = 'E:/openpose/crawler/crop_face/'
sources = ['vcg_CN_1_crop/','veer_CN_1_crop/']
ages = ['baby/','child/','early_youth/','youth/','middle_age/','older/']
genders = ['male/','female/','gender/']
gender_feature=[]
age_feature=[]
for source in sources:
 for age_index,age in enumerate(ages):
  for gender_index,gender in enumerate(genders):
   filedir=file+source+age+gender
   if os.path.exists(filedir):
    filepathes = os.listdir(filedir)

    for impath in filepathes:
            im=cv2.imread(filedir+impath)
  #          img_gray = cv2.cvtColor(im, cv2.IMREAD_GRAYSCALE)
            Hog_feature = My_Hog(im)
            age_feature.append([np.transpose(Hog_feature)[0][:],age_index])
            if (age_index >1):
              gender_feature.append([np.transpose(Hog_feature)[0][:],gender_index])
            pd.DataFrame(age_feature).to_csv('HOG_age.txt',header=None,index=None)
            pd.DataFrame(gender_feature).to_csv('HOG_gender.txt',header=None,index=None)
#pca = PCA(n_components=2)
#Hog_feature=pca.fit_transform(feature)        
#        sift = cv2.xfeatures2d.SIFT_create()
#        keypoints, descriptor = sift.detectAndCompute(img_gray, None)
#        img = cv2.drawKeypoints(image=im, outImage=im, keypoints=keypoints,
#                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
#                          color=(51, 163, 236))
# 
#        
#        winSize = (64,64)
#        blockSize = (16,16)
#        blockStride = (8,8)
#        cellSize = (8,8)
#        nbins = 9
#        derivAperture = 1
#        winSigma = 4.
#        histogramNormType = 0
#        L2HysThreshold = 2.0000000000000001e-01
#        gammaCorrection = 0
#        nlevels = 64
#        hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
#                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
#        #compute(img[, winStride[, padding[, locations]]]) -> descriptors
#        winStride = (8,8)
#        padding = (8,8)
#        locations = ((10,20),)
#        hist = hog.compute(im,winStride,padding,locations)
#        print(hist.shape)