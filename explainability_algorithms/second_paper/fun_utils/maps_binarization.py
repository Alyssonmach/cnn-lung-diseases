import explainability_algorithms as ea
import occlusion_algorithms as oa
import matplotlib.pyplot as plt
import numpy as np
import cv2

def extract_annotations(img_path, target_size):
  '''
  binarize the segmentation images

  Args:
    img_path (str) --> directory path where the image is located
    target_size (list) --> image dimension to add to array
  
  Returns:
    img_binarized (array) --> binarized image
  '''

  img = plt.imread(img_path)
  img = cv2.resize(src = img, dsize = target_size, interpolation = cv2.INTER_NEAREST)
  pet_region = (img * 255.) == 1.
  pet_shadow = (img * 255.) == 3.
  
  return pet_region + pet_shadow

def extract_gradcam(img_array, last_conv_layer_name, copy_model):
  '''
  binarize gradcam activation maps

  Args:
    img_array (array) --> array of the original reference image
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    copy_model (tensorflow model) --> copy of compiled model architecture
  
  Returns:
    map_binarized (array) --> binarized image
  '''
  
  map, _ = ea.get_grad_cam(img_array = img_array, last_conv_layer_name = last_conv_layer_name, 
                           copy_model = copy_model)
  
  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
  
  return map_gray > np.mean(map_gray)

def extract_backpropagation(img_array, model, copy_model, last_conv_layer_name, type):
  '''
  binarize guided backpropagation algorithms 

  Args:
    img_array (array) --> array of the original reference image
    model (tensorflow model) --> trained tensorflow model
    copy_model (tensorflow model) --> copy of compiled model architecture
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    type (str) --> type of algorithm used
  
  Returns:
    map_binarized (array) --> binarized image
  '''

  if type == 'guided_backpropagation':
    map = ea.guided_backpropagation(img_array = img_array, model = model, 
                                    last_conv_layer_name = last_conv_layer_name)  
  elif type == 'guided_gradcam':
    map = ea.guided_grad_cam(img_array = img_array, last_conv_layer_name = last_conv_layer_name, 
                             model = model, copy_model = copy_model)

  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)

  return map_gray > np.mean(map_gray) + np.std(map_gray)

def extract_occlusion(img_array, patch_size, channels, patches_dims, model, val_occlusion):
  '''
  returns a binarized mask of the occlusion prediction algorithm

  Args:
    img_array (array) --> array of the original reference image
    patch_size (int) --> size of an individual patch
    channels (int) --> number of channels in the image
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
    model (tensorflow model) --> trained tensorflow model
  
  Returns:
    img_binarized (array) --> binarized image
  '''

  map = oa.occlusion_prediction(img_array = img_array, patch_size = patch_size, 
                                  channels = channels, patches_dims = patches_dims, 
                                  model = model, val_occlusion = val_occlusion)
  
  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
  
  return map_gray > 0

def extract_gradcam_otsu(img_array, last_conv_layer_name, copy_model):
  '''
  binarize grad cam maps using otsu algorithm

  Args:
    img_array (array) --> array of the original reference image
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    copy_model (tensorflow model) --> copy of compiled model architecture
  
  Returns:
    map_binarized (array) --> binarized image 
  '''

  map, _ = ea.get_grad_cam(img_array = img_array, last_conv_layer_name = last_conv_layer_name, 
                           copy_model = copy_model)
  
  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
  
  _, thresh = cv2.threshold((map_gray * 255).astype('uint8'), 0, 1, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU) 

  return thresh > 0

def extract_backpropagation_otsu(img_array, model, copy_model, last_conv_layer_name, type):
  '''
  binarize guided backpropagation maps using otsu algorithm

  Args:
    img_array (array) --> array of the original reference image
    model (tensorflow model) --> trained tensorflow model
    copy_model (tensorflow model) --> copy of compiled model architecture
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    type (str) --> type of algorithm used
  
  Returns:
    map_binarized (array) --> binarized image
  '''

  if type == 'guided_backpropagation':
    map = ea.guided_backpropagation(img_array = img_array, model = model, 
                                    last_conv_layer_name = last_conv_layer_name)  
  elif type == 'guided_gradcam':
    map = ea.guided_grad_cam(img_array = img_array, last_conv_layer_name = last_conv_layer_name, 
                             model = model, copy_model = copy_model)
  
  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)

  _, thresh = cv2.threshold((map_gray * 255).astype('uint8'), 0, 1, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  return thresh > 0

def extract_occlusion_otsu(img_array, patch_size, channels, patches_dims, model, val_occlusion):
  '''
  binarize occlusion predicts maps using otsu algorithm

  Args:
    img_array (array) --> array of the original reference image
    patch_size (int) --> size of an individual patch
    channels (int) --> number of channels in the image
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
    model (tensorflow model) --> trained tensorflow model
  
  Returns:
    img_binarized (array) --> binarized image
  '''

  map = oa.occlusion_prediction(img_array = img_array, patch_size = patch_size, 
                               channels = channels, patches_dims = patches_dims, 
                               model = model, val_occlusion = val_occlusion)
  
  map_gray = cv2.cvtColor(map, cv2.COLOR_RGB2GRAY)
  
  _, thresh = cv2.threshold((map_gray * 255).astype('uint8'), 0, 1, 
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  
  return thresh > 0