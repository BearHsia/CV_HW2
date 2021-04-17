from PIL import Image
import numpy as np

def get_tiny_images(image_paths):
    #############################################################################
    # TODO:                                                                     #
    # To build a tiny image feature, simply resize the original image to a very #
    # small square resolution, e.g. 16x16. You can either resize the images to  #
    # square while ignoring their aspect ratio or you can crop the center       #
    # square portion out of each image. Making the tiny images zero mean and    #
    # unit length (normalizing them) will increase performance modestly.        #
    #############################################################################
    '''
    Input : 
        image_paths: a list(N) of string where each string is an image 
        path on the filesystem.
    Output :
        tiny image features : (N, d) matrix of resized and then vectorized tiny
        images. E.g. if the images are resized to 16x16, d would equal 256.
    '''
    img = Image.open(image_paths[0])
    img = img.resize((20, 20))
    tiny_images = np.array(img)
    tiny_images = np.reshape(tiny_images,(-1,400))

    for i in range(1,len(image_paths)):
        img = Image.open(image_paths[i])
        img = img.resize((20, 20))
        tmp = np.array(img)
        tmp = np.reshape(tmp,(-1,400))
        tiny_images = np.concatenate((tiny_images, tmp), axis=0)

    avg_of_rows = tiny_images.sum(axis=1)/400
    tiny_images = tiny_images - avg_of_rows[:,np.newaxis]
    
    sqrt_sum_of_rows = np.sqrt(np.square(tiny_images).sum(axis=1))
    tiny_images = tiny_images/sqrt_sum_of_rows[:,np.newaxis]

    ##############################################################################
    #                                END OF YOUR CODE                            #
    ##############################################################################
    return tiny_images
