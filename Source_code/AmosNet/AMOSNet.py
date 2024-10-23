from __future__ import division
import caffe
import numpy as np
import os
import glob
import cv2
import tifffile as tiff
from time import time
import math
from os.path import dirname
import shutil

mean_npy = np.load(str(os.path.abspath(os.curdir))+'/AmosNet/amosnet_mean.npy') # Input numpy array
print('Mean Array Shape:' + str(mean_npy.shape))    #Mean Array Shape:(3, 256, 256)
mean_npy = np.resize(mean_npy,(3,227,227))
net = caffe.Net(str(os.path.abspath(os.curdir))+'/AmosNet/deploy.prototxt', str(os.path.abspath(os.curdir))+'/AmosNet/AmosNet.caffemodel', caffe.TEST)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mean_npy)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

def compute_query_desc(image_query):
    

    features_query_local=np.zeros((256,30))

    image_query = image_query / 255.
    image_query = image_query[:,:,(2,1,0)]
    
    if (image_query is not None):
        
        transformed_image_query = transformer.preprocess('data', image_query)
        net.blobs['data'].data[...] = transformed_image_query.copy()
        out = net.forward()
        features_query=np.asarray(net.blobs['conv5'].data)[1,:,:,:].copy()

        features_query_local=np.zeros((256,30))

        for i in range(256):

                #S=1
                features_query_local[i,0]=np.max(features_query[i,:,:])

                #S=2


                features_query_local[i,1]=np.max(features_query[i,0:6,0:6])
                features_query_local[i,2]=np.max(features_query[i,0:6,7:12])
                features_query_local[i,3]=np.max(features_query[i,7:12,0:6])
                features_query_local[i,4]=np.max(features_query[i,7:12,7:12])

                #S=3

                features_query_local[i,5]=np.max(features_query[i,0:4,0:4])
                features_query_local[i,6]=np.max(features_query[i,0:4,5:8])
                features_query_local[i,7]=np.max(features_query[i,0:4,9:12])
                features_query_local[i,8]=np.max(features_query[i,5:8,0:4])
                features_query_local[i,9]=np.max(features_query[i,5:8,5:8])
                features_query_local[i,10]=np.max(features_query[i,5:8,9:12])
                features_query_local[i,11]=np.max(features_query[i,9:12,0:4])
                features_query_local[i,12]=np.max(features_query[i,9:12,5:8])
                features_query_local[i,13]=np.max(features_query[i,9:12,9:12])

                #S=4
                features_query_local[i,14]=np.max(features_query[i,0:3,0:3])
                features_query_local[i,15]=np.max(features_query[i,0:3,4:6])
                features_query_local[i,16]=np.max(features_query[i,0:3,7:9])
                features_query_local[i,17]=np.max(features_query[i,0:3,10:12])
                features_query_local[i,18]=np.max(features_query[i,4:6,0:3])
                features_query_local[i,19]=np.max(features_query[i,4:6,4:6])
                features_query_local[i,20]=np.max(features_query[i,4:6,7:9])
                features_query_local[i,21]=np.max(features_query[i,4:6,10:12])
                features_query_local[i,22]=np.max(features_query[i,7:9,0:3])
                features_query_local[i,23]=np.max(features_query[i,7:9,4:6])
                features_query_local[i,24]=np.max(features_query[i,7:9,7:9])
                features_query_local[i,25]=np.max(features_query[i,7:9,10:12])
                features_query_local[i,26]=np.max(features_query[i,10:12,0:3])
                features_query_local[i,27]=np.max(features_query[i,10:12,4:6])
                features_query_local[i,28]=np.max(features_query[i,10:12,7:9])
                features_query_local[i,29]=np.max(features_query[i,10:12,10:12])

    return features_query_local


query_dir = '{query_img_dir}/'
ref_dir = '/{ref_img_dir}/'
query_images_names = [os.path.basename(x) for x in glob.glob(query_dir + '*.jpg')]
ref_images_names = [os.path.basename(x) for x in glob.glob(ref_dir + '*.jpg')]
print (len(query_images_names))
print (len(ref_images_names))
similarity_martix = np.zeros((len(query_images_names),len(ref_images_names)))
np.store_flat_query = np.zeros((1,256*30))
np.store_flat_ref = np.zeros((1,256*30))

def cosSim(x, y):
    # cos_sim
    tmp = sum(a * b for a, b in zip(x, y))
    non = np.linalg.norm(x) * np.linalg.norm(y)
    return round(tmp / float(non), 6)

def extr_feature():

    i = 0
    j = 0
    for img_name in sorted(query_images_names):
        if i < len(query_images_names):
            # query_img = tiff.imread(query_dir2 + img_name)  # If the image is saved as a .tif file, use this line of code
            query_img = cv2.imread(query_dir + img_name)  # If the image is saved as a .jpg file, use this line of code
            query_img2 = cv2.resize(query_img, (227, 227))
            extra_query = compute_query_desc(query_img2)
            np.flat_query_feature = extra_query.flatten()
            new_row_1 = np.flat_query_feature
            np.store_flat_query = np.vstack((np.store_flat_query, new_row_1))
            i = i + 1
    store_flat_query = np.store_flat_query



    for img_name2 in sorted(ref_images_names):
        if j < len(ref_images_names):
            # ref_img = tiff.imread(ref_dir2 + img_name2)  # If the image is saved as a .tif file, use this line of code
            ref_img = cv2.imread(ref_dir + img_name2)  # # If the image is saved as a .jpg file, use this line of code
            ref_img2 = cv2.resize(ref_img, (227, 227))
            extra_ref = compute_query_desc(ref_img2)
            np.flat_ref_feature = extra_ref.flatten()
            new_row_2 = np.flat_ref_feature
            np.store_flat_ref = np.vstack((np.store_flat_ref, new_row_2))
            j = j + 1
    store_flat_ref = np.store_flat_ref

    # calculate similarity
    for i in range(0, len(query_images_names)):
        get_query_feature = store_flat_query[i + 1, :]
        for j in range(0, len(ref_images_names)):
            get_ref_feature = store_flat_ref[j + 1, :]
            similarity_martix[i, j] = cosSim(get_ref_feature, get_query_feature)
        j = 0
    print('-------------------------------------result-------------------------------------')
    print('similarity_martix.shape = ', similarity_martix.shape)
    print('martix = ', similarity_martix)

def write_result():
    with open('/{result_martix_AMOSNet.txt_dir}', 'w') as f:
        for row in similarity_martix:
            format_row = ' '.join(['%.6f' % num for num in row])
            f.write(format_row + '\n')
    print ('The similarity_martix has been successfully written to "result_martix_AMOSNet.txt"!')
    print ('--------------------------------------end---------------------------------------')

# main
extr_feature()
write_result()

        

        
