import numpy as np
import pandas as pd
import cv2
import torch
import albumentations as A
from lungs_segmentation.pre_trained_models import create_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from lungs_segmentation.pre_trained_models import create_model
import lungs_segmentation.inference as inference
import matplotlib.pyplot as plt
import matplotlib

model = create_model("resnet34")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

def segmentation(image, path = '/content/image.png'):

  matplotlib.image.imsave(path, image)
  image, mask = inference.inference(model, path, 0.2)
  #os.remove(path)
  for values_i in range(0, len(mask[0])):
    for values_j in range(0, len(mask[0])):
      if (mask[0, values_i, values_j] + mask[1, values_i, values_j]) == 0:
        image[values_i, values_j, 0] = 0
        image[values_i, values_j, 1] = 0
        image[values_i, values_j, 2] = 0
  
  return image
