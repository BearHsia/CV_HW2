from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def find_max_occurence(K_list):
    occur_dict = {}
    for elem in K_list:
        if elem in occur_dict:
            occur_dict[elem] = occur_dict[elem]+1
        else:
            occur_dict[elem] = 1
    return max(occur_dict, key=(lambda key: occur_dict[key]))
    
def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    K = 11 #for tiny_image feature K=11 or 7
    train_test_dist = distance.cdist(train_image_feats, test_image_feats,'minkowski', p=1) #(N,M)
    distance_index = np.argsort(np.argsort(train_test_dist,axis=0),axis=0)
    test_predicts = []
    for test_i in range(test_image_feats.shape[0]): #0~M
        #test_predicts.append(random.choice(train_labels))
        col_index = distance_index[:,test_i].tolist()
        cls_idx_zip = zip(train_labels,col_index)
        sorted_cls_list = sorted(cls_idx_zip, key=lambda x:x[1])
        sorted_cls_list,_ = zip(*sorted_cls_list)
        sorted_cls_list = list(sorted_cls_list)
        sorted_cls_list = sorted_cls_list[:K]
        test_predicts.append(find_max_occurence(sorted_cls_list))

    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts
