# importing useful packages for building the class
from tf_explain.core.occlusion_sensitivity import OcclusionSensitivity
from tf_explain.core.vanilla_gradients import VanillaGradients
from tf_explain.core.smoothgrad import SmoothGrad
from tf_explain.core import IntegratedGradients
import tensorflow.keras.backend as K
from skimage.transform import resize
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2

def image_preprocessing(path_file, img_size):
  '''
  performs image pre-processing according to network input standards

  Args:
    path_file (str) --> directory path where the image is located
    img_size (list) --> two-dimensional tuple with network input image dimensions
  
  Returns:
    image (array) --> returns an array of the image in the format [1, width, height, channels]
  '''
  image = tf.keras.preprocessing.image.load_img(path = path_file, target_size = img_size) 
  image =  tf.keras.preprocessing.image.img_to_array(image) / 255.
  
  return np.expand_dims(image, axis = 0)

class ExplainabilityAlgorithms():
  ''' 
  modular class with functional explainability algorithms in networks developed 
  using the tensorflow 2.0 framework

  Author: Alysson Machado de Oliveira Barbosa - Federal University of Campina Grande (Brazil)
  '''

  def __init__(self, model, conv_layer_name, img_size, preprocessing_function):
    '''
    class constructor

    Args:
      model (tensor) --> model developed and trained using tensorflow 2.0
      conv_layer_name (str) --> name of the most specialized convolution layer in 
      extracting definitive features for classification
      img_size (tuple) --> input image dimensions for the network
      preprocessing_function (function) --> function that performs preprocessing 
      on the image following network standards
    '''

    # OBSERVAÇÃO: TRATAR OS MAPAS DE SALIÊNCIA COM BINARIZAÇÃO?
    # OBSERVAÇÃO: TESTAR POSSIBILIDADES DE JUNÇÃO DOS ALGORITMOS EM UM NOVO MAPA

    self.model = model
    self.conv_layer_name = conv_layer_name
    self.img_size = img_size
    self.preprocessing_function = preprocessing_function

  # modularly alters the gradient return using a decorator to eliminate negative gradients
  @tf.custom_gradient
  def guidedRelu(self, x):
    def grad(dy):
      return tf.cast(dy > 0, "float32") * tf.cast(x > 0, "float32") * dy
    return tf.nn.relu(x), grad
    
  def get_grads(self, path_file):
    '''
    get the analysis image gradients

    Args:
      path_file (str) --> directory path where the image is located
    
    Returns:
      tape_gradients (tensor image) --> scaled gradients in the analysis image
      ouputs (tensor list) --> scaled gradients in the last convolution layer
    '''

    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)

    # creating a model whose final layer is the specified convolutional layer
    gb_model = tf.keras.models.Model(inputs = [self.model.inputs], 
                                     outputs = [self.model.get_layer(self.conv_layer_name).output])
    layer_dict = [layer for layer in gb_model.layers[1:] if hasattr(layer, 'activation')]

    # apply guided relu to all the convolutional layers where activation was relu
    for layer in layer_dict:
      if layer.activation == tf.keras.activations.relu: layer.activation = self.guidedRelu

    # finding the specific gradients in the image under analysis
    with tf.GradientTape() as tape:
      inputs = tf.cast(preprocessed_image, tf.float32)
      tape.watch(inputs)
      outputs = gb_model(inputs)[0]
    
    return tape.gradient(outputs, inputs)[0], outputs

  # OBSERVAÇÃO 1: ORGANIZAR AQUI A UTILIZAÇÃO DE GRADIENTES ESPECÍFICOS POR CLASSE
  # OBSERVAÇÃO 2: TORNAR MAIS INTUITIVO AS CORES NO MAPA DE SALIÊNCIA GERADO
  def grad_cam(self, path_file):
    '''
    generates a salience map based on the grad cam algorithm

    Args:
      path_file (str) --> directory path where the image is located
    
    Returns:
      grad_cam_img (array) --> salience map generated in the dimensions of the convolutional 
      layer of analysis, generated through cubic interpolation
      grad_cam_plot (array) --> salience map generated in the dimensions of the original image, 
      generated through cubic interpolation
    '''

    grads, outputs = self.get_grads(path_file = path_file)
    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)

    # average the gradients spatially where each entry is the mean intensity of 
    # the gradient over a specific feature map channel
    weights = tf.reduce_mean(grads, axis=(0, 1))
    grad_cam = np.ones(outputs.shape[0: 2], dtype = np.float32)
    # build a ponderated map of filters according to gradient importance 
    # the result is the final class discriminative saliency map
    for i, w in enumerate(weights): grad_cam += w * outputs[:, :, i]
    
    grad_cam_img = cv2.resize(src = grad_cam.numpy(), dsize = self.img_size, interpolation = cv2.INTER_CUBIC)
    grad_cam_img = np.maximum(grad_cam_img, 0)
    heatmap = (grad_cam_img - grad_cam_img.min()) / (grad_cam_img.max() - grad_cam_img.min())
    grad_cam_img = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_INFERNO)
    grad_cam_plot = cv2.addWeighted(src1 = cv2.cvtColor(preprocessed_image[0].astype('uint8'), cv2.COLOR_RGB2BGR), 
                                    alpha = 0.5, src2 = grad_cam_img, beta = 1, gamma = 0)
    
    return grad_cam, grad_cam_plot
  
  def guided_backpropagation(self, path_file):
    '''
    generates a salience map based on the guided backpropagatrion algorithm

    Args:
      path_file (str) --> directory path where the image is located
    
    Returns:
      guided_gb (array) --> salience map generated in the dimensions of the original image
    '''

    grads, _ = self.get_grads(path_file = path_file)

    guided_back_prop = grads
    guided_gb = np.dstack((guided_back_prop[:, :, 0], guided_back_prop[:, :, 1],
                           guided_back_prop[:, :, 2]))       
    guided_gb -= np.min(guided_gb)
    guided_gb /= guided_gb.max()

    return guided_gb
  
  def guided_grad_cam(self, path_file):
    '''
    generates a salience map based on the guided grad cam algorithm

    Args:
      path_file (str) --> directory path where the image is located
    
    Returns:
      guided_gc (array) --> salience map generated in the dimensions of the original image
    '''

    guided_back_prop, _ = self.get_grads(path_file = path_file)
    grad_cam, _ = self.grad_cam(path_file = path_file)

    guided_cam = np.maximum(grad_cam, 0)
    guided_cam = guided_cam / np.max(guided_cam) 
    guided_cam = resize(guided_cam, self.img_size, preserve_range = True)
    guided_gc = np.dstack((guided_back_prop[:, :, 0] * guided_cam, guided_back_prop[:, :, 1] * guided_cam,
                           guided_back_prop[:, :, 2] * guided_cam))

    return guided_gc

  # OBSERVAÇÃO 1: TORNAR MAIS INTUITIVO A VISUALIZAÇÃO NO MAPA DE SALIÊNCIA
  # OBSERVAÇÃO 2: UTILIZAR UM OUTRO ALGORITMO PARA MELHORAR O MAPA?
  def occlusion_sensitivity(self, path_file, class_index, patch_size = 7):
    '''
    generates a salience map based on the occlusion sensivity algorithm

    Args:
      path_file (str) --> directory path where the image is located
      class_index (int) --> image analysis class
      patch_size (int) --> sets occlusion kernel size
    
    Returns:
      os_image (array) --> salience map generated in the dimensions of the original image
      os_map (array) --> salience map generated in the dimensions of the original image
    '''

    explainer = OcclusionSensitivity()
    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)
    os_image = explainer.explain(validation_data = (preprocessed_image, None), model = self.model, 
                               class_index = class_index, patch_size = patch_size)
    os_map = explainer.get_sensitivity_map(model = self.model, image = preprocessed_image[0], 
                                           class_index = class_index, patch_size = patch_size)
    
    return os_image, os_map
  
  def smooth_grad(self, path_file, class_index, num_samples = 80, noise = 0.2):
    '''
    generates a salience map based on the smooth grad algorithm

    Args:
      path_file (str) --> directory path where the image is located
      class_index (int) --> image analysis class
      num_samples (int) --> number of example images to be generated
      noise (float) --> probability of gaussian noise generated in the image
    
    Returns:
      sg_map (array) --> salience map generated in the dimensions of the original image
    '''

    explainer = SmoothGrad()
    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)
    sg_map = explainer.explain(validation_data = ([preprocessed_image[0]], None), model = self.model, 
                               class_index = class_index, num_samples = num_samples, noise = noise)
    
    return sg_map
  
  # OBSERVAÇÃO: BINARIZAR OS MAPAS PARA MELHORAR O USO EM POSTERIORES TRATAMENTOS
  def vanilla_gradients(self, path_file, class_index):
    '''
    generates a salience map based on the vanilla gradients algorithm

    Args:
      path_file (str) --> directory path where the image is located
      class_index (int) --> image analysis class
    
    Returns:
      vg_map (array) --> salience map generated in the dimensions of the original image
    '''

    explainer = VanillaGradients()
    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)
    vg_map = explainer.explain(validation_data = ([preprocessed_image[0]], None), model = self.model, 
                               class_index = class_index)
    
    return vg_map
  
  # OBSERVAÇÃO: BINARIZAR OS MAPAS PARA MELHORAR O USO EM POSTERIORES TRATAMENTOS
  def integrated_gradients(self, path_file, class_index):
    '''
    generates a salience map based on the integrated gradients algorithm

    Args:
      path_file (str) --> directory path where the image is located
      class_index (int) --> image analysis class
    
    Returns:
      ig_map (array) --> salience map generated in the dimensions of the original image
    '''

    explainer = IntegratedGradients()
    preprocessed_image = self.preprocessing_function(path_file = path_file, img_size = self.img_size)
    ig_map = explainer.explain(validation_data = ([preprocessed_image[0]], None), model = self.model, 
                               class_index = class_index)
    
    return ig_map