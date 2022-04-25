import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import random
import cv2

def image_patches(img_array, patch_size, channels):
  '''
  separate images by patches
  
  Args:
    img_array (array) --> array of the original reference image
    patch_size (int) --> size of an individual patch
    channels (int) --> number of channels in the image
  
  Returns:
    img_patches (array) --> separate image in several patches
  '''

  patches = tf.image.extract_patches(
    images = img_array, sizes = [1, patch_size, patch_size, 1],
    strides = [1, patch_size, patch_size, 1], rates = [1, 1, 1, 1], 
    padding = 'VALID')
  
  patch_dims = patches.shape[-1]
  patches = tf.reshape(tensor = patches, shape = [1, -1, patch_dims])

  return np.resize(patches[0], (patches.shape[1], patch_size, patch_size, channels))

def reconstructed_image(patches, patches_dims):
  '''
  rebuild the image from your patches

  Args:
    patches (array) -->image separated by patches
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
  
  Returns:
    reconstructed_image (array) --> image reconstructed by patches
  '''

  count  = 0
  for i in range(0, patches_dims[0]):
    for j in range(0, patches_dims[1]):

      if j == 0:
        reconstruction_h = patches[count]
      elif j != 0:
        reconstruction_h = np.hstack((reconstruction_h, patches[count]))

      count += 1
    
    if i == 0:
      reconstruction_v = reconstruction_h 
    elif i != 0:
      reconstruction_v = np.vstack((reconstruction_v, reconstruction_h))
  
  return reconstruction_v

def make_occlusion(patches, patches_dims, val_occlusion = 128):
  '''
  algorithm that performs occlusion on image patches

  Args:
    patches (array) -->image separated by patches
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
    val_occlusion (float) --> occlusion pixel value

  Returns:
    samples_images (array) --> occluded image samples
  '''

  images = list()
  occlusion_patch = np.ones(patches[0].shape) * val_occlusion

  for i in range(0, len(patches)):
    copy_patches = np.array(patches)
    copy_patches[i] = occlusion_patch 
    images.append(reconstructed_image(patches = copy_patches, patches_dims = patches_dims))
  
  return np.array(images)

def occlusion_prediction(img_array, patch_size, channels, patches_dims, model, val_occlusion = 128):
  '''
  predicts the image using the occlusion algorithm

  Args:
    img_array (array) --> array of the original reference image
    patch_size (int) --> size of an individual patch
    channels (int) --> number of channels in the image
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
    model (tensorflow model) --> trained tensorflow model
    val_occlusion (float) --> occlusion pixel value

  Returns:
    image_occlusion (array) --> predicted image occlusions
  '''

  pat = image_patches(img_array = img_array, patch_size = patch_size, channels = channels)
  imgs = make_occlusion(patches = pat, patches_dims = patches_dims, 
                        val_occlusion = val_occlusion)
  predict_max = np.argmax(model.predict(img_array))

  predictions = list()
  for img in imgs:
    predictions.append(model.predict(tf.expand_dims(img, axis = 0)))

  mean_value = np.mean(np.array(predictions)[:,0][:, predict_max])
  predictions = np.array(predictions)[:,0]

  for index, predict in enumerate(predictions):
    if predict[predict_max] > mean_value:
      pat[index] = val_occlusion

  image_occlusion = reconstructed_image(patches = pat, patches_dims = (4, 4))

  return image_occlusion / np.max(image_occlusion)

def occlusion_prediction_plot(img_array, patch_size, channels, patches_dims, model, val_occlusion = 128):
  '''
  predicts the image using the occlusion algorithm

  Args:
    img_array (array) --> array of the original reference image
    patch_size (int) --> size of an individual patch
    channels (int) --> number of channels in the image
    patchs_dims (list) -->list containing (number of patches vertically, 
    number of patches horizontally)
    model (tensorflow model) --> trained tensorflow model
    val_occlusion (float) --> occlusion pixel value

  Returns:
    image_occlusion (array) --> predicted image occlusions
  '''

  pat = image_patches(img_array = img_array, patch_size = patch_size, 
                      channels = channels)
  imgs = make_occlusion(patches = pat, patches_dims = patches_dims, 
                        val_occlusion = val_occlusion)
  predict_max = np.argmax(model.predict(img_array))

  predictions = list()
  for img in imgs:
    predictions.append(model.predict(tf.expand_dims(img, axis = 0)))

  mean_value = np.mean(np.array(predictions)[:,0][:, predict_max])
  predictions = np.array(predictions)[:,0]

  pred_min = list()
  for index, predict in enumerate(predictions):
    if predict[predict_max] < mean_value:
      pred_min.append([index, predict[predict_max]])
  pred_min = np.array(pred_min)

  list_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0],
                 [255, 0, 255], [0 ,255, 255], [128, 0, 0], [0, 128, 0], 
                 [0, 0, 128], [128, 128, 0], [128, 0, 128], [0 ,128, 128],
                 [255, 50, 50], [50, 255, 50], [50, 50, 255], [255, 255, 50],
                 [255, 50, 255], [50 ,255, 255], [128, 50, 50], [50, 128, 50], 
                 [50, 50, 128], [128, 128, 50], [128, 50, 128], [50 ,128, 128]]
  random.shuffle(list_colors)

  fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (10, 4))
  x = np.arange(1, len(predictions[:, predict_max]) + 1)
  axs[1].plot(x, np.ones(x.shape) * np.round(mean_value, 2), linestyle = '--', )
  axs[1].plot(x, np.round(predictions[:, predict_max], 2))
  axs[1].set_title('Predições por Oclusão', size = 15)
  axs[1].set_xlabel('Imagens Ocluídas', size = 10)
  axs[1].set_ylabel('Acurácia da Predição', size = 10)
  axs[1].grid(True), axs[1].set_xlim([1, 15]), axs[1].set_xticks(x)
  
  zeros_pat = np.zeros(pat.shape)
  for index, min in enumerate(pred_min):
    axs[1].scatter(min[0] + 1, np.round(min[1], 2), color = np.array(list_colors[index]) / 255., 
                   alpha = 1)
    pad_shape = zeros_pat[min[0].astype('uint8')].shape
    pad_with_border = cv2.copyMakeBorder(zeros_pat[min[0].astype('uint8')], 2, 2, 2, 2, 
                                         cv2.BORDER_CONSTANT, value = list_colors[index])
    pad_with_border = cv2.resize(pad_with_border, pad_shape[:2], 
                                 interpolation = cv2.INTER_NEAREST)
    zeros_pat[min[0].astype('uint8')] = pad_with_border

  sel_pat = reconstructed_image(patches = zeros_pat, patches_dims = (4, 4))
  img_with_patches = img_array[0] * 0.3 + (sel_pat / np.max(sel_pat))
  axs[0].imshow(img_with_patches / np.max(img_with_patches))
  axs[0].set_title('Patches Selecionados', size = 15), axs[0].axis('off')
  
  return 