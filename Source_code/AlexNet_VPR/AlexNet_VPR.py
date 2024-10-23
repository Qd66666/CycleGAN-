import glob

import caffe
import numpy as np
import cv2
import tifffile as tiff
from time import time
import os
import math
from os.path import dirname
import shutil

first_it = True
A = None


alexnet_proto_path = "/{deploy.prototxt_dir}"
alexnet_weights = "/{bvlc_alexnet.caffemodel_dir}"

alexnet = caffe.Net(alexnet_proto_path, 1, weights=alexnet_weights)

transformer_alex = caffe.io.Transformer({'data': (1, 3, 227, 227)})
transformer_alex.set_raw_scale('data',1./255)
transformer_alex.set_transpose('data', (2, 0, 1))
transformer_alex.set_channel_swap('data', (2, 1, 0))

def compute_query_desc(image_query):


    alexnet.blobs['data'].data[...] = transformer_alex.preprocess('data', image_query)

    alexnet.forward()
    alex_conv3 = np.copy(alexnet.blobs['conv3'].data[...])
    alex_conv3 = np.reshape(alex_conv3, (alex_conv3.size, 1))
    global first_it
    global A
    if first_it:
        np.random.seed(0)
        A = np.random.randn(1064, alex_conv3.size) # For Gaussian random projection  descr[0].size=1064
        first_it = False
    alex_conv3 = np.matmul(A, alex_conv3)
    alex_conv3 = np.reshape(alex_conv3, (1, alex_conv3.size))
    alex_conv3 /= np.linalg.norm(alex_conv3)
    
    return alex_conv3


query_dir = '/{query_img_dir}/'
ref_dir = '/{ref_img_dir}/'
query_images_names = [os.path.basename(x) for x in glob.glob(query_dir + '*.jpg')]
ref_images_names = [os.path.basename(x) for x in glob.glob(ref_dir + '*.jpg')]
print (len(query_images_names))  #439
print (len(ref_images_names))    #1571

similarity_martix = np.zeros((len(query_images_names),len(ref_images_names)))

np.store_flat_query = np.zeros((1,1064))
np.store_flat_ref = np.zeros((1,1064))



def extr_feature():

    i = 0
    j = 0

    for img_name in sorted(query_images_names):
        if i < len(query_images_names):
            query_img = cv2.imread(query_dir + img_name)
            query_img2 = cv2.resize(query_img, (227, 227))
            query_img3 = cv2.cvtColor(query_img2, cv2.COLOR_GRAY2BGR)
            extra_query = compute_query_desc(query_img3)
            np.flat_query_feature = extra_query.flatten()
            new_row_1 = np.flat_query_feature
            np.store_flat_query = np.vstack((np.store_flat_query, new_row_1))
            i = i + 1
    store_flat_query = np.store_flat_query

    for img_name2 in sorted(ref_images_names):
        if j < len(ref_images_names):
            ref_img = cv2.imread(ref_dir + img_name2)
            ref_img2 = cv2.resize(ref_img, (227,227))
            ref_img3 = cv2.cvtColor(ref_img2, cv2.COLOR_GRAY2BGR)
            extra_ref = compute_query_desc(ref_img3)
            np.flat_ref_feature = extra_ref.flatten()
            new_row_2 = np.flat_ref_feature
            np.store_flat_ref = np.vstack((np.store_flat_ref, new_row_2))

            j = j + 1
    store_flat_ref = np.store_flat_ref


# main
extr_feature()