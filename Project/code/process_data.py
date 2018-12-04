import numpy as np
import glob
from skimage import color
from skimage import io
import cv2

def image_manipulating(input):
    """
   Randomly output the same image but having diffrent angle.

    """
    mode=np.random.randint(0,8)
    if mode == 0:
        return input
    elif mode == 1:
        return np.flipud(input)
    elif mode == 2:
        return np.rot90(input)
    elif mode == 3:
        return np.flipud(np.rot90(input))
    elif mode == 4:
        return np.rot90(input, k=2)
    elif mode == 5:
        return np.flipud(np.rot90(input, k=2))
    elif mode == 6:
        return np.rot90(input, k=3)
    elif mode == 7:
        return np.flipud(np.rot90(input, k=3))

def process_data(url_data,h,w, stride = 10):
    
    """
    Create the training data for model based on undamaged images.    
    Arguments:
    url_data  -- The path of images 
    For example the url_data like url_data = '/floyd/input/rawdata/*.png'
    h: the height of images
    w: the weight of images
    stride: number of jummping step

    Returns:
    The training images data.
    """
    
    image_dir = (url_data)
    image_list_grey = []
    for filename in glob.glob(image_dir): 
        
     # Process Grey Images
      # keep_grey = color.rgb2gray(io.imread(filename))
      keep_grey = cv2.imread(filename, 0)  # gray scale

      h_img, w_img = keep_grey.shape

      for i  in range (0, h_img - h + 1, stride):
          for j in range(0, w_img - w + 1, stride):
              tmp_img = keep_grey[i:i + h, j:j + w]
              image_list_grey.append(tmp_img)
              for k in range (0,7):
                  tmp_img_manipulate = image_manipulating(tmp_img)
                  if ( np.array_equal(tmp_img_manipulate, tmp_img) == False ):
                      image_list_grey.append(tmp_img_manipulate)
      image_list_grey = np.array(image_list_grey, dtype='uint8')
      print('Finish process data  and the data shape : is ' + str(image_list_grey.shape))
      return image_list_grey


def train_datagen(data,epoch_num = 2000, batch_size = 16, set_epoch = False, process = 'Training Proccess'):
      """
      Creating the training and validation batch data.
      Arguments:
      data: The whole images data need to seperate into small batch
      epoch_num:the estimating steeps for process data
      batch_size: the size of batch

      Returns:
      set of noise images, unnoise images
      """
      if set_epoch == False:
          epoch_num = int(max(data.shape)/batch_size)
          print(process + ', set_epoch: ' + str(epoch_num))


      while(True):
        n_count = 0
        if n_count == 0:
            print('Data Normalization')
            print('Data Shape' + str(data.shape))
            data = data.astype('float32')/255.0
            indices = list(range(data.shape[0]))
            n_count = 1
        for _ in range(epoch_num):
            np.random.shuffle(indices)    # shuffle
            for i in range(0, len(indices), batch_size):
                if (i%200 ==0):
                    print('i = : ' + str(i))
                batch_x = data[indices[i:i+batch_size]]
                noise =  np.random.normal(0, 25/255.0, batch_x.shape)    # noise
                batch_y = batch_x + noise
                batch_y = batch_y.reshape((batch_y.shape[0],batch_y.shape[1],batch_y.shape[2],1))
                batch_x = batch_x.reshape((batch_x.shape[0],batch_x.shape[1],batch_x.shape[2],1))
                yield batch_y, batch_x

