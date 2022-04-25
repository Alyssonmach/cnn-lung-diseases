from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference
import matplotlib.pyplot as plt
import skimage.morphology
import tensorflow as tf
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib
import random
import heapq
import torch
import cv2

class bbox_utils():

  def __init__(self) -> None:
      pass

  def model_segmentation(self, architecture = 'resnet34'):
    '''
    instantiates the lung segmentation model from the 'lungs segmentation' package

    Args:
      architecture (str) --> name of the base canonical architecture of the model
    
    Return:
      model (object) --> lung segmentation model 
    '''

    try:
      model = create_model(architecture)
    except:
      print('network architecture not available, configuring resnet34...')
      model = create_model('resnet34')
    
    return model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

  def make_segmentation(self, model, path_image):
    '''
    performs lung segmentation on an image
    
    Args:
      path_image (str) --> destination file of the image to be targeted

    Returns:
      image (array) --> image in original dimensions
      mask (list) --> list of arrays with left and right lung segmentation masks
    '''

    image = plt.imread(path_image)
    width, height = image.shape[0], image.shape[1] 

    image, mask = inference.inference(model = model, img_path = path_image, thresh = 0.2)

    image = cv2.resize(src = image, dsize = (height, width), interpolation = cv2.INTER_NEAREST, )
    mask_1 = cv2.resize(src = mask[0], dsize = (height, width), interpolation = cv2.INTER_NEAREST)
    mask_2 = cv2.resize(src = mask[1], dsize = (height, width), interpolation = cv2.INTER_NEAREST)

    return image, np.array([mask_1, mask_2])

  def join_masks(self, lung_left, lung_right):
    '''
    joins the left lung and right lung masks

    Args:
      lung1 (array) --> left lung segmentation mask
      lung2 (array) --> right lung segmentation mask

    Returns:
      mask (array) --> left and right lung segmentation mask
    '''

    return np.array(lung_left + lung_right)

  def get_contours(self, mask, erode_radius = 10):
    '''
    get the outlines of the elements in the segmentation mask
    
    Args:
      mask (array) --> segmentation mask array 
      erode_radius (int) --> structuring element radius size 

    Returns:
      contours (list) --> array list of mask element outlines  
    '''
    
    mask = cv2.morphologyEx(src = mask.copy(), op = cv2.MORPH_ERODE, kernel = skimage.morphology.disk(radius = erode_radius))
    ret, thresh = cv2.threshold(src = mask, thresh = 0.5, maxval = 1, type = cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(image = thresh, mode = cv2.RETR_TREE, method = cv2.CHAIN_APPROX_SIMPLE)

    return contours

  def get_contour_areas(self, contours):
    '''
    get the boundary areas through a list

    Args:
      contours (list) --> list with the boundary regions of the delimited elements

    Returns:
      all_areas (list) --> list with the boundary areas of the delimited elements
    '''

    all_areas = list()

    for cnt in contours:
      area = cv2.contourArea(contour = cnt)
      all_areas.append(area)

    return all_areas

  def find_two_best_areas(self, contours):
    '''
    find the two best contour areas

    Args:
      contours (list) --> list with the boundary areas of the delimited elements
    
    Returns:
      contours (list) --> list with the two largest outline areas of the delimited elements
    '''

    list_areas = self.get_contour_areas(contours = contours)

    if len(list_areas) >= 2:
      two_elements = heapq.nlargest(n = 2, iterable = list_areas)
      return [contours[list_areas.index(two_elements[0])], contours[list_areas.index(two_elements[1])]]
    
    return contours  

  def draw_contours(self, contours, image_shape, dilate_radius = 6):
    '''
    creates a segmentation mask using the obtained contour regions  

    Args:
      contours (list) --> list of bypassed regions
      image_shape (tuple) --> tuple with the original two dimensions of the image
      dilate_radius (int) --> structuring element radius size 
    
    Returns:
      mask (array) --> segmentation mask based on obtained contours
    '''

    mask = cv2.drawContours(image = np.zeros(shape = image_shape) , contours = self.find_two_best_areas(contours), 
                            contourIdx = -1, color = (0, 255, 0), thickness = cv2.FILLED)
    _, mask = cv2.threshold(src = mask, thresh = 127, maxval = 1, type = 0)
    mask = cv2.morphologyEx(src = mask, op = cv2.MORPH_DILATE, kernel = skimage.morphology.disk(radius = dilate_radius))

    return mask  

  def resize_image(self, image, shape):
    '''
    resizes the image to specific dimensions 

    Args:
      image (array) --> image with original dimensions
      shape (tuple) --> image dimensions after resizing
    
    Returns:
      image (array) --> resized image
    '''

    return cv2.resize(src = image, dsize = shape, interpolation = cv2.INTER_NEAREST)

  def dot_set_p1(self, mask, dim_max, axis = 0):
    '''
    computes the first points of the bounding box

    Args:
      mask (array) --> segmentation mask
      dim_max (int) --> maximum dimension of the reference coordinate
      axis (int) --> delimits the axes (x, y) to obtain the coordinate of the bounding box

    Returns:
      p1 (int) --> coordinate of the first points of the bounding box
    '''

    p1 = 0
    for index in range(0, dim_max):
      if (np.max(mask[index, :]) == 1 and not(axis)) or (np.max(mask[:, index]) == 1 and axis):
        p1 = index
        break
      
    return p1

  def dot_set_p2(self, mask, dim_max, axis = 0):
    '''
    computes the second points of the bounding box

    Args:
      mask (array) --> segmentation mask
      dim_max (int) --> maximum dimension of the reference coordinate
      axis (int) --> delimits the axes (x, y) to obtain the coordinate of the bounding box

    Returns:
      p1 (int) --> coordinate of the second points of the bounding box
    '''

    p2 = dim_max
    for index in range(1, dim_max + 1):
      if (np.max(mask[-index, :]) == 1 and not(axis)) or (np.max(mask[:, -index]) == 1 and axis):
        p2 = abs(index - dim_max)
        break
      
    return p2

  def make_percentual_coordinates(self, list_coordinates, shape):
    '''
    transforms pixel coordinates to percentage coordinates

    Args:
      list_coordinates (list) --> list of coordinates in relative pixels
      shape (tuple) --> image dimension

    Returns:
      list_coordinates (list) --> list of percentage coordinates of the bounding box 
    '''

    list_coordinates[0] = list_coordinates[0] / shape[0]
    list_coordinates[1] = list_coordinates[1] / shape[1]
    list_coordinates[2] = list_coordinates[2] / shape[0]
    list_coordinates[3] = list_coordinates[3] / shape[1]

    return list_coordinates

  def add_percentual_margin(self, percentual_coordinates, shape, list_of_margins = [8, 8, 20, 8]):
    '''
    adds a margin of error in the bounding boxes

    Args:
      percentual_coordinates (list) --> list of bounding box percentage coordinates  
      shape (tuple) --> image dimension 
      list_of_margins (list) --> margin list in relative pixels

    Returns:
      percentual_coordinates (list) --> list of percentage coordinates of the bounding box with margin of error
    '''

    percentual_coordinates[0] = percentual_coordinates[0] - (list_of_margins[0] / shape[0])
    percentual_coordinates[1] = percentual_coordinates[1] - (list_of_margins[1] / shape[1])
    percentual_coordinates[2] = percentual_coordinates[2] + (list_of_margins[2] / shape[0])
    percentual_coordinates[3] = percentual_coordinates[3] + (list_of_margins[3] / shape[1])

    for index, value in enumerate(percentual_coordinates):
      if index <= 1:
        if percentual_coordinates[index] < 0: percentual_coordinates[index] = 0.0
      else:
        if percentual_coordinates[index] > 1: percentual_coordinates[index] = 1.0

    return percentual_coordinates    

  def extract_coordinates(self, mask, list_of_margins = [8, 8, 20, 8]):
    '''
    extract the coordinates of the buffer's bounding box from the segmentation mask

    Args:
      mask (array) --> segmentation mask 
      list_of_margins (list) --> list of bounding box margins
    Returns:
      coordinates (list) --> list of coordinates that delimit the lung region  
    '''
    mask = self.resize_image(image = mask, shape = (320, 320))
    width, height = mask.shape[0], mask.shape[1]
    p1_x = self.dot_set_p1(mask = mask, dim_max = width, axis = 0)
    p1_y = self.dot_set_p1(mask = mask, dim_max = height, axis = 1)
    p2_x = self.dot_set_p2(mask = mask, dim_max = width, axis = 0)
    p2_y = self.dot_set_p2(mask = mask, dim_max = height, axis = 1)
    percentual_coordinates = self.make_percentual_coordinates(list_coordinates = [p1_x, p1_y, p2_x, p2_y], shape = mask.shape)
    percentual_coordinates = self.add_percentual_margin(percentual_coordinates = percentual_coordinates, shape = mask.shape, list_of_margins = list_of_margins)

    return np.array(percentual_coordinates)

  def draw_rectangle(self, image, list_coordinates, color = (255, 0, 0), thickness = 2):
    '''
    draws a rectangle around the lung based on the coordinates

    Args:
      image (array) --> radiography image
      list_coordinates (list) --> list of rectangle percentage coordinates 

    Returns:
      image (array) --> image with localized lung
    '''

    width, height = image.shape[0], image.shape[1]

    list_coordinates = [int(list_coordinates[0] * width), int(list_coordinates[1] * height),
                        int(list_coordinates[2] * width), int(list_coordinates[3] * height)]

    image_out = cv2.rectangle(img = image.copy(), pt1 = (list_coordinates[1], list_coordinates[0]),
                              pt2 = (list_coordinates[3], list_coordinates[2]), color = color,
                              thickness = thickness)
    
    return image_out

  def crop_image(self, image, list_coordinates):
    '''
    cuts the image from the bounding box

    Args:
      image (array) --> radiography image
      list_coordinates (list) --> list of radiograph coordinates

    Returns:
      image (array) --> x-ray lung region  
    '''

    width, height = image.shape[0], image.shape[1]
    
    list_coordinates = [int(list_coordinates[0] * width) - 4,
                        int(list_coordinates[1] * height) - 4,
                        int(list_coordinates[2] * width) + 4,
                        int(list_coordinates[3] * height) + 4]
    
    if list_coordinates[0] < 0: list_coordinates[0] = 0
    if list_coordinates[1] < 0: list_coordinates[1] = 0
    if list_coordinates[2] > width: list_coordinates[2] = width
    if list_coordinates[3] > height: list_coordinates[3] = height
    
    try:
      img_crop = image[list_coordinates[0]:list_coordinates[2], list_coordinates[1]:list_coordinates[3]]
      return img_crop
    except:
      return image

  def save_img(self, image, path_file):
    '''
    save the image from an array

    Args:
      image (array) --> radiography image
      path_file (str) --> path where the image is located
      name (str) --> image file name 
      format (str) --> image format to be saved 
    '''

    return plt.imsave(path_file, image)

  def make_localization_cnn(self, model, path_file):
    '''
    obtains the bounding box from a convolutional neural network model of tensorflow

    Args:
      model (tensor) --> tensorflow convolutional neural network model
      path_files (str) --> image destination to be inferred by the algorithm

    Returns:
      list_coordinates (list) --> list of bounding box coordinates
    '''

    image = cv2.imread(path_file)
    image = cv2.resize(src = image, dsize = (128, 128), interpolation = cv2.INTER_NEAREST) / 255.
    list_coordinates = model.predict(np.reshape(image, newshape = (1, 128, 128, 3)))

    return list_coordinates[0]