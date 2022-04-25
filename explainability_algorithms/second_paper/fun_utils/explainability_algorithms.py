from skimage.transform import resize
import matplotlib.cm as cm
import tensorflow as tf
from copy import copy
import numpy as np
import cv2

def copy_model(model):
  '''
  creates a copy of the model made by tensorflow
  
  Args:
    model (model tensorflow) --> compiled model architecture

  Returns:
    copy_model (model tensorflow) --> copy of compiled model architecture
  '''

  return copy(model)

def predicted_best_class(img_array, model):
  '''
  returns the index of the class predicted by the model
  
  Args:
    img_array (array) --> array of the image to be predicted by the model
    model (model tensorflow) --> model trained with tensorflow
  
  Returns:
    argmax (int) --> maximum argument index of the predicted class
  '''

  return np.argmax(model.predict(img_array))

def get_img_array(img_path, target_size):
  '''
  returns an image array in standard tensorflow input format

  Args:
    img_path (str) --> file location where the image is saved
    target_size (list) --> image dimension to add to array

  Returns:
    img_array (array) --> image in array format
  '''
    
  try: 
    img = tf.keras.preprocessing.image.load_img(path = img_path, target_size = target_size)
  except: 
    img = img_path

  img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.

  return np.expand_dims(a = img_array, axis = 0)

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index = None, 
                         auto_index = True):
  '''
  build a gradcam activation map
  
  Args:
    img_array (array) --> reference image for building the activation map
    model (model tensorflow) --> trained tensorflow model
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    pred_index (int) --> target class index
    auto_index (bool) --> if true, use the best class predicted by the model

  Returns:
    gradcam_map (array) --> activation map made with gradcam algorithm
  '''
  
  if auto_index:
    pred_index = predicted_best_class(img_array = img_array, model = model)
  input = [model.inputs]
  output = [model.get_layer(last_conv_layer_name).output, model.output]
  grad_model = tf.keras.models.Model(inputs = input, outputs = output)

  with tf.GradientTape() as tape:
    last_conv_layer_output, preds = grad_model(img_array)
    if pred_index is None:
      pred_index = tf.argmax(preds[0])
    class_channel = preds[:, pred_index]

  grads = tape.gradient(class_channel, last_conv_layer_output)
  pooled_grads = tf.reduce_mean(grads, axis = (0, 1, 2))
  last_conv_layer_output = last_conv_layer_output[0]
  heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
  heatmap = tf.squeeze(heatmap)
    
  return (tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)).numpy()

def resize_gradcam(img_array, heatmap):
  '''
  resize grad cam class activation map to original image dimensions

  Args:
    img_array (array) --> array of the original reference image
    heatmap (array) --> built activation map array

  Returns:
    heatmap_resized (array) --> activation map with the same dimensions as the 
    original image
  '''
   
  heatmap = np.uint8(255 * heatmap)
  jet_colors = cm.get_cmap("jet")(np.arange(256))[:, :3]
  jet_heatmap = jet_colors[heatmap]

  jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
  jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    
  return tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
def get_grad_cam(img_array, last_conv_layer_name, copy_model, 
                 alpha = 0.9, pred_index = None, auto_index = True):
  '''
    builds class activation map using gradcam algorithm

    Args:
      img_array (array) --> array of the original reference image
      last_conv_layer_name (str) --> name of the last convolution layer of the model
      copy_model (model tensorflow) --> copy of compiled model architecture
      alpha (float) --> multiplicative factor of visualization of the activation map 
      superimposed on the image
      pred_index (int) --> target class index
      auto_index (bool) --> if true, use the best class predicted by the model
      
    Returns:
      gradcam_image (array) --> gradcam activation map
      overlap_image (array) --> gradcam activation map overlaid on image
  '''

  copy_model.layers[-1].activation = None
  heatmap = make_gradcam_heatmap(img_array = img_array, model = copy_model, 
                                 last_conv_layer_name = last_conv_layer_name,
                                 pred_index = pred_index, auto_index = auto_index)
    
  gradcam_image = resize_gradcam(img_array = img_array[0], heatmap = heatmap) / 255.
  overlap_image = img_array[0] + gradcam_image * alpha

  return gradcam_image / np.max(gradcam_image), overlap_image / np.max(overlap_image)

@tf.custom_gradient
def guidedRelu(x):
  '''
  activation function based on the backpropagation gradient

  Args:
    x (float) --> activation function input data
  Returns:
    x_activation (float) --> x output with relu activation function
    x_grad (float) --> gradient x calculation
  '''

  def grad(dy):

    return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy

  return tf.nn.relu(x), grad

def get_grads(img_array, model, last_conv_layer_name):
  '''
  get the analysis image gradients

  Args:
    img_array (array) --> array of the original reference image
    model (tensorflow model) --> trained tensorflow model
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    
  Returns:
    tape_gradients (tensor image) --> scaled gradients in the analysis image
    ouputs (tensor list) --> scaled gradients in the last convolution layer
  '''

  input = [model.inputs]
  output = [model.get_layer(last_conv_layer_name).output]
  gb_model = tf.keras.models.Model(inputs = input, outputs = output)
  layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]

  for layer in layer_dict:
    if layer.activation == tf.keras.activations.relu: layer.activation = guidedRelu

  with tf.GradientTape() as tape:
    inputs = tf.cast(img_array, tf.float32)
    tape.watch(inputs)
    outputs = gb_model(inputs)[0]
    
  return tape.gradient(outputs, inputs)[0], outputs

def guided_backpropagation(img_array, model, last_conv_layer_name):
  '''
  generates a salience map based on the guided backpropagatrion algorithm

  Args:
    img_array (array) --> array of the original reference image
    model (tensorflow model) --> trained tensorflow model
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    
  Returns:
    guided_gb (array) --> salience map generated in the dimensions of the original image
  '''

  grads, _ = get_grads(img_array = img_array, model = model, 
                       last_conv_layer_name = last_conv_layer_name, )

  guided_back_prop = grads
  guided_gb = np.dstack((guided_back_prop[:, :, 0], 
                         guided_back_prop[:, :, 1],
                         guided_back_prop[:, :, 2]))       
  guided_gb -= np.min(guided_gb)
  guided_gb /= guided_gb.max()

  return (guided_gb) / np.max(guided_gb)

def guided_grad_cam(img_array, last_conv_layer_name, model, copy_model, 
                    alpha = 0.9, pred_index = None, auto_index = True):
  '''
  generates a salience map based on the guided grad cam algorithm

  Args:
    img_array (array) --> array of the original reference image
    last_conv_layer_name (str) --> name of the last convolution layer of the model
    model (tensorflow model) --> trained tensorflow model
    copy_model (tensorflow model) --> copy of compiled model architecture
    alpha (float) --> multiplicative factor of visualization of the activation map 
    superimposed on the image
    pred_index (int) --> target class index
    auto_index (bool) --> if true, use the best class predicted by the model
    
  Returns:
    guided_gc (array) --> salience map generated in the dimensions of the original image
  '''

  grad_cam, _ = get_grad_cam(img_array = img_array, 
                             last_conv_layer_name = last_conv_layer_name, 
                             copy_model = copy_model, alpha = alpha, 
                             pred_index = pred_index, auto_index = auto_index)
  guided_back_prop, _ = get_grads(img_array = img_array, model = model, 
                                  last_conv_layer_name = last_conv_layer_name, )

  guided_cam = cv2.cvtColor(grad_cam, cv2.COLOR_RGB2GRAY)
  guided_cam = np.maximum(guided_cam, 0.5)
  guided_cam = guided_cam / np.max(guided_cam) 
  guided_gc = np.dstack((guided_back_prop[:, :, 0] * guided_cam, 
                         guided_back_prop[:, :, 1] * guided_cam,
                         guided_back_prop[:, :, 2] * guided_cam))
  
  guided_gc = np.maximum(guided_gc, 0)

  return (guided_gc) / np.max(guided_gc)
