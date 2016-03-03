#!/usr/bin/env python

import sys, math, Image, os
import csv

def Distance(p1,p2):
  dx = p2[0] - p1[0]
  dy = p2[1] - p1[1]
  return math.sqrt(dx*dx+dy*dy)

def ScaleRotateTranslate(image, angle, center = None, new_center = None, scale = None, resample=Image.BICUBIC):
  if (scale is None) and (center is None):
    return image.rotate(angle=angle, resample=resample)
  nx,ny = x,y = center
  sx=sy=1.0
  if new_center:
    (nx,ny) = new_center
  if scale:
    (sx,sy) = (scale, scale)
  cosine = math.cos(angle)
  sine = math.sin(angle)
  a = cosine/sx
  b = sine/sx
  c = x-nx*a-ny*b
  d = -sine/sy
  e = cosine/sy
  f = y-nx*d-ny*e
  return image.transform(image.size, Image.AFFINE, (a,b,c,d,e,f), resample=resample)

def CropFace(image, eye_left=(0,0), eye_right=(0,0), offset_pct=(0.2,0.2), dest_sz = (70,70)):
  # calculate offsets in original image
  offset_h = math.floor(float(offset_pct[0])*dest_sz[0])
  offset_v = math.floor(float(offset_pct[1])*dest_sz[1])
  # get the direction
  eye_direction = (eye_right[0] - eye_left[0], eye_right[1] - eye_left[1])
  # calc rotation angle in radians
  rotation = -math.atan2(float(eye_direction[1]),float(eye_direction[0]))
  # distance between them
  dist = Distance(eye_left, eye_right)
  # calculate the reference eye-width
  reference = dest_sz[0] - 2.0*offset_h
  # scale factor
  scale = float(dist)/float(reference)
  # rotate original around the left eye
  image = ScaleRotateTranslate(image, center=eye_left, angle=rotation)
  # crop the rotated image
  crop_xy = (eye_left[0] - scale*offset_h, eye_left[1] - scale*offset_v)
  crop_size = (dest_sz[0]*scale, dest_sz[1]*scale)
  image = image.crop((int(crop_xy[0]), int(crop_xy[1]), int(crop_xy[0]+crop_size[0]), int(crop_xy[1]+crop_size[1])))
  # resize it
  image = image.resize(dest_sz, Image.ANTIALIAS)
  return image

if __name__ == "__main__":
  path = './Faces3/'
  new_path = './Faces2/'

  with open('rotate.csv', 'rb') as csvfile:
    data = csv.reader(csvfile, delimiter=',')
    dict1 = {rows[0]:[int(rows[1]),int(rows[2]),int(rows[3]),int(rows[4])] for rows in data}
  
  image_paths = [os.path.join(path, f) for f in os.listdir(path)]
  for image_path in image_paths:
    image =  Image.open(image_path)
    fileName = image_path.split('/')[2]
    fileName = fileName.split('.')[0]
    res = CropFace(image, eye_left=(dict1[fileName][0],dict1[fileName][1]), eye_right=(dict1[fileName][2],dict1[fileName][3]), offset_pct=(0.4,0.4), dest_sz=(640,480))
    res.save(new_path+fileName+'.jpg')

    #CropFace(image, eye_left=(dict1[fileName][0],dict1[fileName][1]), eye_right=(dict1[fileName][2],dict1[fileName][3]), offset_pct=(0.1,0.1), dest_sz=(200,200)).save(new_path+fileName+'.jpg')
    #CropFace(image, eye_left=(dict1[fileName][0],dict1[fileName][1]), eye_right=(dict1[fileName][2],dict1[fileName][3]), offset_pct=(0.2,0.2), dest_sz=(200,200)).save(new_path+fileName+'.jpg')
    #CropFace(image, eye_left=(dict1[fileName][0],dict1[fileName][1]), eye_right=(dict1[fileName][2],dict1[fileName][3]), offset_pct=(0.2,0.2)).save(new_path+fileName+'.jpg')
