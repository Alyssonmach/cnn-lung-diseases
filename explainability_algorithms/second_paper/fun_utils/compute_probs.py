def count_pixels_intersection(image_segmentation, image_map):
  '''
  counts the number of pixels 1's that intersect two images

  Args:
    image_segmentation (array) --> binarized object segmentation map
    image_map (array) --> binarized object class activation map
  
  Returns:
    count (int) --> number of pixels 1's that intersect two images
  '''

  flatten_seg, flatten_map = image_segmentation.flatten(), image_map.flatten()

  count = 0
  for i in range(0, len(flatten_seg)):
    if flatten_seg[i] == flatten_map[i] and (flatten_seg[i] != 0):
      count += 1

  return count

def count_pixels_1s(image):
  '''
  counts the number of pixels with value 1 in an binarized image

  Args:
    image (array) --> binarized image
  
  Returns
    count (int) --> number of pixels 1's present in the image
  '''

  flatten_image = image.flatten()

  count = 0
  for i in range(0, len(flatten_image)):
    if flatten_image[i] == 1:
      count += 1
  
  return count

def intersect_seg_map_prob(image_segmentation, image_map):
  '''
  probability of intersection of two binarized images

  Args:
    image_segmentation (array) --> binarized object segmentation map
    image_map (array) --> binarized object class activation map
    
  Returns:
    count (int) --> number of pixels 1's that intersect the two images
  '''

  num = count_pixels_intersection(image_segmentation, image_map)
  den = count_pixels_1s(image_map)

  if den == 0: return 0
  else: return num / den
