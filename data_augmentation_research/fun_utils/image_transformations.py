import random as rd
import numpy as np
import cv2

class geometric_transformations():

  def __init__(self) -> None:
      pass

  def sp_noise(self, image, prob = 0.02, rate = 0.5):
    '''
    adds salt and pepper noise to the image

    Args:
      image (array) --> array of the image to be transformed
      prob (int) --> probability of adding noise to the image
      rate (float) --> rate of increase of adjustable parameters
    
    Returns:
      image (array) --> geometrically transformed image
    '''
  
    image_output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        rdn = rd.random()
        if rdn < (prob * rate):
          image_output[i][j] = 0
        elif rdn > 1 - (prob * rate):
          image_output[i][j] = 255
        else:
          image_output[i][j] = image[i][j]
                  
    return image_output

  def gaussian_noise(self, image, prob = 0.02, rate = 0.5):
    '''
    adds gaussian noise to the image

    Args:
      image (array) --> array of the image to be transformed
      prob (int) --> probability of adding noise to the image
      rate (float) --> rate of increase of adjustable parameters
    
    Returns:
      image (array) --> geometrically transformed image
    '''
  
    image_output = np.zeros(image.shape, np.uint8)
    for i in range(image.shape[0]):
      for j in range(image.shape[1]):
        rdn = abs(np.random.normal(0, 0.1))
        if rdn < (prob * rate):
          image_output[i][j] = 0
        elif rdn > 1 - (prob * rate):
          image_output[i][j] = 255
        else:
          image_output[i][j] = image[i][j]
                  
    return image_output

  def rotate_image(self, image, angle = 20, rate = 0.5):
    '''
    apply clockwise and counterclockwise rotations to the image

    Args:
      image (array) --> array of the image to be transformed
      angle (int) --> image rotation angle
      rate (float) --> rate of increase of adjustable parameters

    Returns:
      image (array) --> geometrically transformed image
    '''

    direction = rd.randrange(0, 2)
    if direction == 1: angle *= -1

    rot_mat = cv2.getRotationMatrix2D(tuple(np.array(image.shape) / 2), angle * rate, 1.0)

    return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_NEAREST)

  def image_translation(self, image, translate_lim = [45, 45], rate = 0.5):
    '''
    translates the image on the abscissa and ordinate axes

    Args:
      image (array) --> image to be transformed geometrically
      translate_lim (list) --> x-axis and y-axis translation limits
      rate (float) --> rate of increase of adjustable parameters

    Returns:
      image (array) --> geometrically transformed image
    '''

    direction1, direction2 = rd.randrange(0, 2), rd.randrange(0, 2)
    if direction1 == 1: translate_lim[0] *= -1
    if direction2 == 1: translate_lim[1] *= -1

    width, height = image.shape[0], image.shape[1]
    matrix = np.array([[1, 0, int(translate_lim[0] * rate)], [0, 1, int(translate_lim[1] * rate)]], dtype = np.float32)
    
    return cv2.warpAffine(image, matrix, (height, width))

class filters_transformations():

  def __init__(self) -> None:
      pass

  def gamma_correction(self, image, gamma = 2.0, rate = 0.5):
    '''
    apply gamma correction to the image

    Args:
      image (array) --> image to be filtered
      gamma --> gamma correction rate
      rate (float) --> rate of increase of adjustable parameters

    Returns:
      image (array) --> filtered image
    '''
    
    table = np.array([((i / 255.0) ** (1.0 / gamma * rate)) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    return cv2.LUT(image, table)

  def log_transformation(self, image):
    '''
    apply logarithmic transformation to the image

    Args:
      image (array) --> image to be filtered

    Returns:
      image (array) --> filtered image
    '''
    
    return np.array((255 / np.log(1.0 + np.max(image))) * (np.log(image + 1.0)), dtype = np.uint8)

  def adaptative_histogram_equalization(self, image):
    '''
    applies an adaptive histogram correction to the image

    Args:
      image (array) --> image to be filtered

    Returns:
      image (array) --> filtered image
    '''

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    try:
      return clahe.apply(image) 
    except:
      return clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)) 

  def mean_filter(self, image, kernel = (2, 2)):
    '''
    apply the anti-aliasing filter to the image

    Args:
      image (array) --> image to be filtered
      kernel (list) --> kernel dimensions

    Returns:
      image (array) --> filtered image
    '''
  
    return cv2.filter2D(src = image, ddepth = -1, kernel = np.ones(kernel,np.float32) / (kernel[0] * kernel[1]))

  def median_filter(self, image, nsize = 3):
    '''
    apply the median smoothing filter to the image

    Args:
      image (array) --> image to be filtered
      nsize (int) --> kernel size

    Returns:
      image (array) --> filtered image
    '''

    return cv2.medianBlur(src = image, ksize = nsize)

  def gaussian_filter(self, image, size = (3,3), sigma = 10):
    '''
    apply the Gaussian smoothing filter

    Args:
      image (array) --> image to be filtered
      size (list) --> kernel dimension
      sigma (int) --> sigma constant of the Gaussian operation

    Returns:
      image (array) --> filtered image
    '''

    return cv2.GaussianBlur(src = image, ksize = size, sigmaX = sigma)

  def sharpening(self, image):
    '''
    apply the sharpening filter to the image

    Args:
      image (array) --> image to be filtered

    Returns:
      image (array) --> filtered image
    '''
    
    Gx = abs(cv2.filter2D(src = image, ddepth = -1, kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])))
    Gy = abs(cv2.filter2D(src = image, ddepth = -1, kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])))

    return (Gx + Gy).astype(np.uint8)

class image_manipulation():

  def __init__(self) -> None:
      pass
  
  def organize_image(self, path_image, dsize = (256, 256)):
    '''
    organizes the analysis images
    
    Args:
      path_image (str) --> image relative path
      dsize (list) --> new dimension of the image
    
    Returns:
      image (numpy) --> numpy array with image content
    '''

    image = cv2.imread(filename = path_image)
    image = cv2.cvtColor(image, code = cv2.COLOR_BGR2GRAY)
    image = cv2.resize(src = image, dsize = dsize, interpolation = cv2.INTER_NEAREST)

    return image