import numpy as np
import glob
from skimage import color
from skimage import io

def process_data(url_data,h,w,seed):
    
    """
    Create the training data for model based on undamaged images.    
    Arguments:
    url_data  -- The path of images 
    For example the url_data like url_data = '/floyd/input/rawdata/*.png'
    h: the height of images
    w: the weight of images
    seed: the random seed for generating noise images

    Returns:
    The noise images, unnoise images and residual images. 
    """
    
    image_dir = (url_data)
    image_list_grey = []
    image_list_grey_noise = []
    image_list_residual_error = []
    mean = 0
    std = 25/255.0
    np.random.seed(9001)
    for filename in glob.glob(image_dir): 
        
     # Process Grey Images
      keep_grey = color.rgb2gray(io.imread(filename))
      if (int(keep_grey.shape[0]) >= h and int(keep_grey.shape[1]) >= w):
          
          keep_grey = np.array(keep_grey)[0:h,0:w]
          image_list_grey.append(keep_grey)
          keep_grey = []
         
    image_list_grey = np.asarray(image_list_grey)
    image_list_grey_noise = image_list_grey + np.random.normal(mean, std, ((h,w)))
    image_list_residual_error = image_list_grey_noise - image_list_grey
   
    return (image_list_grey,image_list_grey_noise,image_list_residual_error)