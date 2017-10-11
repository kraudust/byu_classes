import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import io as skio
from skimage import transform
import random
from pdb import set_trace as stop
import pickle

#------------------------------------------------Load Data ---------------------------------------------------
#Load all filenames into a lists
pos_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/pos')
pos_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/pos')
neg_train_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/train/neg')
neg_train_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/train/neg')
pos_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/pos')
pos_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/pos')
neg_test_filenames = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/inputs/test/neg')
neg_test_labels  = os.listdir('/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/outputs/test/neg')


n = 300 #number of training images to use
train_index = random.sample(range(len(pos_train_filenames)),n) #randomly pick indices for images to train
im_size = 512 #resize image to im_size x im_size x 3
#create variables to hold the resized images and labels
train_ims = np.zeros((2*n,im_size,im_size,3)).astype(np.float32)
train_labs = np.zeros((2*n,im_size,im_size,1)).astype(np.float32)
test_ims = np.zeros((175,im_size,im_size,3)).astype(np.float32)
test_labs = np.zeros((175, im_size, im_size, 1)).astype(np.float32)
#directory to my data
data_dir = '/home/kraudust/git/personal_git/byu_classes/deep_learning/lab6_cancer_detector/cancer_data/'

#load training data and labels, resize, and put into matrices
for i in xrange(n):
    im_train_pos = skio.imread(data_dir + 'inputs/train/pos/' +   pos_train_filenames[train_index[i]])
    im_lab_pos = skio.imread(data_dir + 'outputs/train/pos/' +   pos_train_labels[train_index[i]])
    im_train_neg = skio.imread(data_dir + 'inputs/train/neg/' +   neg_train_filenames[train_index[i]])
    im_lab_neg = skio.imread(data_dir + 'outputs/train/neg/' +   neg_train_labels[train_index[i]])
    train_ims[2*i,:,:,:] = transform.resize(im_train_pos,(im_size,im_size,3))
    train_ims[2*i+1,:,:,:] = transform.resize(im_train_neg,(im_size,im_size,3))
    train_labs[2*i,:,:,:] = transform.resize(im_lab_pos, (im_size, im_size, 1))
    train_labs[2*i+1,:,:,:] = transform.resize(im_lab_neg, (im_size, im_size, 1))
    print i

#load test data and labels, resize, and put into matrices
for i in xrange(len(neg_test_filenames)):
    if i < 75:
        test_im_pos = skio.imread(data_dir + 'inputs/test/pos/' + pos_test_filenames[i])
        test_im_lab_pos = skio.imread(data_dir + 'outputs/test/pos/' + pos_test_labels[i])
        test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
        test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
        test_ims[2*i,:,:,:] = transform.resize(test_im_pos,(im_size,im_size,3))
        test_ims[2*i+1,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
        test_labs[2*i,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        test_labs[2*i+1,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        print i
    else:
        test_im_neg = skio.imread(data_dir + 'inputs/test/neg/' + neg_test_filenames[i])
        test_im_lab_neg = skio.imread(data_dir + 'outputs/test/neg/' + neg_test_labels[i])
        test_ims[i+75,:,:,:] = transform.resize(test_im_neg,(im_size,im_size,3))
        test_labs[i+75,:,:,:] = transform.resize(test_im_lab_pos, (im_size, im_size, 1))
        print i

# whiten data
train_ims = (train_ims - np.mean(train_ims,0))/(np.std(train_ims,0))
test_ims = (test_ims - np.mean(test_ims,0))/(np.std(test_ims,0))

#---------------------------------------------Pickle Data-----------------------------------------------------
print "Dumping training images..."
pickle.dump(train_ims,open('whitened_training_images','wb'))
print "Dumping training labels..."
pickle.dump(train_labs,open('training_labels','wb'))
print "Dumping test images..."
pickle.dump(test_ims,open('whitened_test_images','wb'))
print "Dumping test labels..."
pickle.dump(test_labs,open('test_labels','wb'))



















stop()
