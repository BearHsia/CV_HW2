from PIL import Image
import numpy as np
from scipy.spatial import distance
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time


def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)
    
    STEP = [2,2]
    FAST = True
    SAMPLE_FEATURE = 1000
    vocab_d = vocab.shape[0]

    def build_histogram(img_,vocab_):
        _, descriptors = dsift(img_,step=STEP,fast=FAST,float_descriptors=True)
        _, descriptors = dsift(img,step=STEP,fast=FAST,float_descriptors=True)
        np.random.shuffle(descriptors)
        descriptors = descriptors[:SAMPLE_FEATURE,:]
        dist = distance.cdist(vocab_, descriptors,'minkowski', p=1) #(N,F)
        argmins = np.argmin(dist,axis=0)
        hist = np.histogram(argmins, bins=range(vocab_d+1),normed=True)[0]
        hist = hist[np.newaxis,:]
        return hist

    img = Image.open(image_paths[0])
    img = np.array(img)
    image_feats = build_histogram(img,vocab)
    for i in range(1,len(image_paths)):
        img = Image.open(image_paths[i])
        img = np.array(img)
        tmp = build_histogram(img,vocab)
        image_feats = np.concatenate((image_feats, tmp), axis=0)
    print(image_feats.shape)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats
