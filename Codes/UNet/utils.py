# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

"""
Functions needed to run the notebooks
"""

"""
Import python packages
"""

import numpy as np
import tensorflow as tf
import skimage

import sys
import os
from scipy import ndimage
from scipy.misc import bytescale
import threading
from threading import Thread, Lock
import h5py

from skimage.io import imread, imsave
import skimage as sk
import tifffile as tiff

import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import random

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard
from keras.optimizers import SGD, RMSprop
    
from datetime import datetime

"""
Helper functions
"""
def extract_channels(input_image_dir, output_image_dir, channels):
    imageFiles = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    os.makedirs(name=output_image_dir, exist_ok=True)

    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(input_image_dir, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = skimage.io.imread(imagePath)
        if image.shape[0]<image.shape[-1]:
            output_image = np.zeros((len(channels), image.shape[1], image.shape[2]), np.uint16)
            for i in range(len(channels)):
                output_image[i, :, :] = (image[channels[i], :, :]).astype('uint16')
            tiff.imsave(os.path.join(output_image_dir, baseName + ".tiff"), output_image)
            
        else:
            output_image = np.zeros((len(channels), image.shape[0], image.shape[1]), np.uint16)
            for i in range(len(channels)):
                output_image[i, :, :] = (image[:, :,channels[i]]).astype('uint16')
            tiff.imsave(os.path.join(output_image_dir, baseName + ".tiff"), output_image)

def divide_images(input_image_dir, output_image_dir, height_divider, width_divider):
    imageFiles = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    os.makedirs(name=output_image_dir, exist_ok=True)

    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(input_image_dir, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = skimage.io.imread(imagePath)
        
        width_channel = 0
        height_channel = 1
        nb_channels = 1
        if len(image.shape)>2:
            if image.shape[0]<image.shape[-1]:
                width = int(image.shape[1]/width_divider)
                height = int(image.shape[2]/height_divider)
                nb_Channels = image.shape[0]
            else:
                width = int(image.shape[0]/width_divider)
                height = int(image.shape[1]/height_divider)
                nb_Channels = image.shape[2]
        else:
            width = int(image.shape[0]/width_divider)
            height = int(image.shape[1]/height_divider)
                
        for i in range(width_divider):
            for j in range(height_divider):
                x_init = int((image.shape[width_channel]/width_divider)*i)
                x_end = x_init + width
                y_init = int((image.shape[height_channel]/height_divider)*j)
                y_end = y_init + height
            
                output_image = np.zeros((nb_channels, width, height), np.uint16)
                if len(image.shape)==2:
                    output_image[0, :, :] = (image[x_init:x_end, y_init:y_end]).astype('uint16')
                    tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                else:
                    if image.shape[0]<image.shape[-1]:
                        output_image = (image[:, x_init:x_end, y_init:y_end]).astype('uint16')
                        tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                    else:
                        output_image = (image[x_init:x_end, y_init:y_end, :]).astype('uint16')
                        tiff.imsave(os.path.join(output_image_dir, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
            

def instance_to_semantic(input_masks_dir, output_masks_dir):
    os.makedirs(name=output_masks_dir, exist_ok=True)

    maskFiles = [f for f in os.listdir(input_masks_dir) if os.path.isfile(os.path.join(input_masks_dir, f))]   
    for index, maskFile in enumerate(maskFiles):
        maskPath = os.path.join(input_masks_dir, maskFile)
        baseName = os.path.splitext(os.path.basename(maskFile))[0]
        mask = skimage.io.imread(maskPath)
    
        neighborhood = [-1, 0, 1]
        if len(mask.shape)>2:
            if mask.shape[0]<mask.shape[-1]:
                mask_UNet = np.zeros((2, mask.shape[1], mask.shape[2]), np.uint16)
                for y in range(mask.shape[2]):
                    for x in range(mask.shape[1]):
                        if mask[0, x, y] > 0:
                            contour = False
                            for v in neighborhood:
                                if y+v >= 0 and y+v < mask.shape[2]:
                                    for u in neighborhood:
                                        if x+u >= 0 and x+u < mask.shape[1]:
                                            if mask[0, x+u, y+v]!=mask[0, x, y]:
                                                contour = True
                            if contour==True:
                                mask_UNet[0, x, y] = 1
                            else:
                                mask_UNet[1, x, y] = 1
            else:
                mask_UNet = np.zeros((2, mask.shape[0], mask.shape[1]), np.uint16)
                for y in range(mask.shape[1]):
                    for x in range(mask.shape[0]):
                        if mask[x, y, 0] > 0:
                            contour = False
                            for v in neighborhood:
                                if y+v >= 0 and y+v < mask.shape[1]:
                                    for u in neighborhood:
                                        if x+u >= 0 and x+u < mask.shape[0]:
                                            if mask[x+u, y+v, 0]!=mask[x, y, 0]:
                                                contour = True
                            if contour==True:
                                mask_UNet[0, x, y] = 1
                            else:
                                mask_UNet[1, x, y] = 1
        else:
            mask_UNet = np.zeros((2, mask.shape[0], mask.shape[1]), np.uint16)
            for y in range(mask.shape[1]):
                for x in range(mask.shape[0]):
                    if mask[x, y] > 0:
                        contour = False
                        for v in neighborhood:
                            if y+v >= 0 and y+v < mask.shape[1]:
                                for u in neighborhood:
                                    if x+u >= 0 and x+u < mask.shape[0]:
                                        if mask[x+u, y+v]!=mask[x, y]:
                                            contour = True
                        if contour==True:
                            mask_UNet[0, x, y] = 1
                        else:
                            mask_UNet[1, x, y] = 1
        
        tiff.imsave(os.path.join(output_masks_dir, baseName + ".tiff"), mask_UNet)

    
def rate_scheduler(lr = .001, decay = 0.95):
    def output_fn(epoch):
        epoch = np.int(epoch)
        new_lr = lr * (decay ** epoch)
        return new_lr
    return output_fn

def process_image(img):
    
    if img.shape[2] == 1:
        output = np.zeros((img.shape[0], img.shape[1], 1), 'float32')
        
        percentile = 99.9
        high = np.percentile(img, percentile)
        low = np.percentile(img, 100-percentile)

        output = np.minimum(high, img)
        output = np.maximum(low, output)

        output = (output - low) / (high - low)

    else:
        output = np.zeros((img.shape[0], img.shape[1], img.shape[2]), 'float32')
        for i in range(img.shape[2]):
            percentile = 99.9
            high = np.percentile(img[:,:,i], percentile)
            low = np.percentile(img[:,:,i], 100-percentile)
            
            output[:,:,i] = np.minimum(high, img[:,:,i])
            output[:,:,i] = np.maximum(low, output[:,:,i])
            
            if high>low:
                output[:,:,i] = (output[:,:,i] - low) / (high - low)

    return output

def getfiles(direc_name):
    imglist = os.listdir(direc_name)
    imgfiles = [i for i in imglist if ('.png'  in i ) or ('.tif' in i) or ('tiff' in i)]

    imgfiles = imgfiles
    return imgfiles

def get_image(file_name):
    if ('.tif' in file_name) or ('tiff' in file_name):
        im = tiff.imread(file_name)
        im = bytescale(im)
        im = np.float32(im)
    else:
        im = np.float32(imread(file_name))
        
    if len(im.shape) < 3:
        output_im = np.zeros((im.shape[0], im.shape[1], 1))
        output_im[:, :, 0] = im
        im = output_im
    else:
        if im.shape[0]<im.shape[2]:
            output_im = np.zeros((im.shape[1], im.shape[2], im.shape[0]))
            for i in range(im.shape[0]):
                output_im[:, :, i] = im[i, :, :]
            im = output_im
    
    return im

"""
Data generator for training_data
"""
def get_data_sample(training_directory, validation_directory, nb_channels = 1, nb_classes = 3, imaging_field_x = 256, imaging_field_y = 256, nb_augmentations = 1, validation_training_ratio = 0.1):

    channels_training = []
    labels_training = []
    channels_validation = []
    labels_validation = []

    imglist_training_directory = os.path.join(training_directory, 'images/')
    masklist_training_directory = os.path.join(training_directory, 'masks/')

    # adding evaluation data into validation
    if validation_directory is not None:

        imglist_validation_directory = os.path.join(validation_directory, 'images/')
        masklist_validation_directory = os.path.join(validation_directory, 'masks/')

        imageValFileList = [f for f in os.listdir(imglist_validation_directory) if os.path.isfile(os.path.join(imglist_validation_directory, f))]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            
            if os.path.exists(os.path.join(masklist_validation_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_validation_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_validation_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_validation_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
            if current_mask_image.shape[0]<imaging_field_x:
                sys.exit("The mask " + baseName + " has a smaller x dimension than the imaging field")
            if current_mask_image.shape[1]<imaging_field_y:
                sys.exit("The mask " + baseName + " has a smaller y dimension than the imaging field")
                
            min_dimension = current_mask_image.shape[2]
            if min_dimension == 1:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image[:, :, 0] == 0, 1, 0)
                for i in range(nb_classes):
                    if i > 0:
                        current_mask[: , :, i-1] = np.where(current_mask_image[:, :, 0] == i, 1, 0)
            else:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                lastClass_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
                for i in range(current_mask_image.shape[2]):
                    current_mask[: , :, i] = np.where(current_mask_image[:, :, i] > 0, 1, 0)
                    lastClass_mask += (current_mask_image[:, :, i]).astype('int32')
                if current_mask_image.shape[2] < nb_classes:
                    current_mask[: , :, (nb_classes-1)] = np.where(lastClass_mask == 0, 1, 0)
                        
            labels_validation.append(current_mask.astype('int32'))
            
            imagePath = os.path.join(imglist_validation_directory, imageFile)
            
            current_image = get_image(imagePath)
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_validation.append(process_image(current_image))

        imageFileList = [f for f in os.listdir(imglist_training_directory) if os.path.isfile(os.path.join(imglist_training_directory, f))]            
        for imageFile in imageFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            
            if os.path.exists(os.path.join(masklist_training_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
            if current_mask_image.shape[0]<imaging_field_x:
                sys.exit("The mask " + baseName + " has a smaller x dimension than the imaging field")
            if current_mask_image.shape[1]<imaging_field_y:
                sys.exit("The mask " + baseName + " has a smaller y dimension than the imaging field")
            
            min_dimension = current_mask_image.shape[2]
            if min_dimension == 1:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image[:, :, 0] == 0, 1, 0)
                for i in range(nb_classes):
                    if i > 0:
                        current_mask[: , :, i-1] = np.where(current_mask_image[:, :, 0] == i, 1, 0)
            else:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                lastClass_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
                for i in range(current_mask_image.shape[2]):
                    current_mask[: , :, i] = np.where(current_mask_image[:, :, i] > 0, 1, 0)
                    lastClass_mask += (current_mask_image[:, :, i]).astype('int32')
                if current_mask_image.shape[2] < nb_classes:
                    current_mask[: , :, (nb_classes-1)] = np.where(lastClass_mask == 0, 1, 0)
            
            labels_training.append(current_mask.astype('int32'))
            
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image(imagePath)
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            channels_training.append(process_image(current_image))
            
    else:
        imageValFileList = [f for f in os.listdir(imglist_training_directory) if os.path.isfile(os.path.join(imglist_training_directory, f))]
        for imageFile in imageValFileList:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imglist_training_directory, imageFile)
            current_image = get_image(imagePath)
            if current_image.shape[0]<imaging_field_x:
                sys.exit("The image " + baseName + " has a smaller x dimension than the imaging field")
            if current_image.shape[1]<imaging_field_y:
                sys.exit("The image " + baseName + " has a smaller y dimension than the imaging field")
            if current_image.shape[2]!=nb_channels:
                sys.exit("The image " + baseName + " has a different number of channels than indicated in the U-Net architecture")
            
            if os.path.exists(os.path.join(masklist_training_directory, baseName + ".png")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".png")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tif")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tif")
            elif os.path.exists(os.path.join(masklist_training_directory, baseName + ".tiff")):
                maskPath = os.path.join(masklist_training_directory, baseName + ".tiff")
            else:
                sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
            current_mask_image = get_image(maskPath)
            if current_mask_image.shape[0]<imaging_field_x:
                sys.exit("The mask " + baseName + " has a smaller x dimension than the imaging field")
            if current_mask_image.shape[1]<imaging_field_y:
                sys.exit("The mask " + baseName + " has a smaller y dimension than the imaging field")

            min_dimension = current_mask_image.shape[2]
            if min_dimension == 1:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                current_mask[: , :, (nb_classes-1)] = np.where(current_mask_image[:, :, 0] == 0, 1, 0)
                for i in range(nb_classes):
                    if i > 0:
                        current_mask[: , :, i-1] = np.where(current_mask_image[:, :, 0] == i, 1, 0)
            else:
                current_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1], nb_classes), 'int32')
                lastClass_mask = np.zeros((current_mask_image.shape[0], current_mask_image.shape[1]), 'int32')
                for i in range(current_mask_image.shape[2]):
                    current_mask[: , :, i] = np.where(current_mask_image[:, :, i] > 0, 1, 0)
                    lastClass_mask += (current_mask_image[:, :, i]).astype('int32')
                if current_mask_image.shape[2] < nb_classes:
                    current_mask[: , :, (nb_classes-1)] = np.where(lastClass_mask == 0, 1, 0)
                    
        
            if random.Random().random() > validation_training_ratio:
                channels_training.append(process_image(current_image))
                labels_training.append(current_mask.astype('int32'))
            else:
                channels_validation.append(process_image(current_image))
                labels_validation.append(current_mask.astype('int32'))

                
    if len(channels_training) < 1:
        sys.exit("Empty train image list")

    #just to be non-empty
    if len(channels_validation) < 1:
        channels_validation += channels_training[len(channels_training)-1]
        labels_validation += channels_validation[len(channels_validation)-1]
    
    
    X_test = channels_validation
    Y_test =[]
    for k in range(len(X_test)):
        X_test[k] = process_image(X_test[k])
        if X_test[k].shape[0]==imaging_field_x:
            start_dim1 = 0
        else:
            start_dim1 = np.random.randint(low=0, high=X_test[k].shape[0] - imaging_field_x)
        if X_test[k].shape[1]==imaging_field_y:
            start_dim2 = 0
        else:
            start_dim2 = np.random.randint(low=0, high=X_test[k].shape[1] - imaging_field_y)
    
        X_test[k] = X_test[k][start_dim1:start_dim1 + imaging_field_x, start_dim2:start_dim2 + imaging_field_y, :]
        Y_test.append(labels_validation[k][start_dim1:start_dim1 + imaging_field_x, start_dim2:start_dim2 + imaging_field_y, :])
        
    train_dict = {"channels": channels_training, "labels": labels_training}

    return train_dict, (np.asarray(X_test).astype('float32'), np.asarray(Y_test).astype('int32'))


def random_sample_generator(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2):

    cpt = 0

    n_images = len(x_init)
    arr = np.arange(n_images)
    np.random.shuffle(arr)
    
    while(True):

        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, n_channels))
        y = np.zeros((batch_size, dim1, dim2, n_classes))
        
        for k in range(batch_size):

            # get random image
            img_index = arr[cpt%n_images]

            # open images
            x_big = x_init[img_index]
            y_big = y_init[img_index]

            # get random crop
            if dim1==x_big.shape[0]:
                start_dim1 = 0
            else:
                start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            if dim2==x_big.shape[1]:
                start_dim2 = 0
            else:
                start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            patch_x = x_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_y = y_big[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_x = np.asarray(patch_x)
            patch_y = np.asarray(patch_y)

            # image normalization
            patch_x = patch_x.astype('float32')

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :, :, :] = patch_y
            cpt += 1

        # return the buffer
        yield(x, y)


def GenerateRandomImgaugAugmentation(
        pAugmentationLevel=5,           # number of augmentations
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=False,           # enable rotation
        pMaxRotationDegree=15,             # maximum rotation degree
        pEnableShearX=False,             # enable x shear
        pEnableShearY=False,             # enable y shear
        pMaxShearDegree=15,             # maximum shear degree
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=.5,                  # maximum sigma for gaussian blur
        pEnableDropOut=True,
        pMaxDropoutPercentage=0.01,
        pEnableSharpness=False,          # enable sharpness
        pSharpnessFactor=0.0001,           # maximum additional sharpness
        pEnableEmboss=False,             # enable emboss
        pEmbossFactor=0.0001,              # maximum emboss
        pEnableBrightness=False,         # enable brightness
        pBrightnessFactor=0.000001,         # maximum +- brightness
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=0.01,           # maximum random noise strength
        pEnableInvert=False,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=0.01,            # maximum +- contrast
):
    
    augmentationMap = []
    augmentationMapOutput = []


    if pEnableFlipping1:
        aug = iaa.Fliplr()
        augmentationMap.append(aug)
        
    if pEnableFlipping2:
        aug = iaa.Flipud()
        augmentationMap.append(aug)

    if pEnableRotation90:
        randomNumber = random.Random().randint(1,3)
        aug = iaa.Rot90(randomNumber)
        augmentationMap.append(aug)

    if pEnableRotation:
        if random.Random().randint(0, 1)==1:
            randomRotation = random.Random().random()*pMaxRotationDegree
        else:
            randomRotation = -random.Random().random()*pMaxRotationDegree
        aug = iaa.Rotate(randomRotation)
        augmentationMap.append(aug)

    if pEnableShearX:
        if random.Random().randint(0, 1)==1:
            randomShearingX = random.Random().random()*pMaxShearDegree
        else:
            randomShearingX = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearX(randomShearingX)
        augmentationMap.append(aug)

    if pEnableShearY:
        if random.Random().randint(0, 1)==1:
            randomShearingY = random.Random().random()*pMaxShearDegree
        else:
            randomShearingY = -random.Random().random()*pMaxShearDegree
        aug = iaa.ShearY(randomShearingY)
        augmentationMap.append(aug)

    if pEnableDropOut:
        randomDropOut = random.Random().random()*pMaxDropoutPercentage
        aug = iaa.Dropout(p=randomDropOut, per_channel=False)
        augmentationMap.append(aug)

    if pEnableBlur:
        randomBlur = random.Random().random()*pBlurSigma
        aug = iaa.GaussianBlur(randomBlur)
        augmentationMap.append(aug)

    if pEnableSharpness:
        randomSharpness = random.Random().random()*pSharpnessFactor
        aug = iaa.Sharpen(randomSharpness)
        augmentationMap.append(aug)

    if pEnableEmboss:
        randomEmboss = random.Random().random()*pEmbossFactor
        aug = iaa.Emboss(randomEmboss)
        augmentationMap.append(aug)

    if pEnableBrightness:
        if random.Random().randint(0, 1)==1:
            randomBrightness = 1 - random.Random().random()*pBrightnessFactor
        else:
            randomBrightness = 1 + random.Random().random()*pBrightnessFactor
        aug = iaa.Add(randomBrightness)
        augmentationMap.append(aug)

    if pEnableRandomNoise:
        if random.Random().randint(0, 1)==1:
            randomNoise = 1 - random.Random().random()*pMaxRandomNoise
        else:
            randomNoise = 1 + random.Random().random()*pMaxRandomNoise
        aug = iaa.MultiplyElementwise(randomNoise,  per_channel=True)
        augmentationMap.append(aug)
        
    if pEnableInvert:
        aug = iaa.Invert(1)
        augmentationMap.append(aug)

    if pEnableContrast:
        if random.Random().randint(0, 1)==1:
            randomContrast = 1 - random.Random().random()*pContrastFactor
        else:
            randomContrast = 1 + random.Random().random()*pContrastFactor
        aug = iaa.contrast.LinearContrast(randomContrast)
        augmentationMap.append(aug)

    arr = np.arange(len(augmentationMap))
    np.random.shuffle(arr)
    for i in range(pAugmentationLevel):
        augmentationMapOutput.append(augmentationMap[arr[i]])
    
        
    return iaa.Sequential(augmentationMapOutput)


def random_sample_generator_dataAugmentation(x_init, y_init, batch_size, n_channels, n_classes, dim1, dim2, nb_augmentations):

    cpt = 0
    n_images = len(x_init)
    arr = np.arange(n_images)
    np.random.shuffle(arr)
    non_augmented_array = np.zeros(n_images)

    while(True):

        # buffers for a batch of data
        x = np.zeros((batch_size, dim1, dim2, n_channels))
        y = np.zeros((batch_size, dim1, dim2, n_classes))
        
        for k in range(batch_size):
            
            # get random image
            img_index = arr[cpt%n_images]

            # open images
            x_big = x_init[img_index]
            y_big = y_init[img_index]

            # augmentation
            segmap = SegmentationMapsOnImage(y_big, shape=x_big.shape)
            augmentationMap = GenerateRandomImgaugAugmentation()

            x_aug, segmap = augmentationMap(image=x_big.astype('float32'), segmentation_maps=segmap)
            y_aug = segmap.get_arr()
            
            # image normalization
            x_norm = x_aug.astype('float32')

            # get random crop
            if dim1==x_big.shape[0]:
                start_dim1 = 0
            else:
                start_dim1 = np.random.randint(low=0, high=x_big.shape[0] - dim1)
            if dim2==x_big.shape[1]:
                start_dim2 = 0
            else:
                start_dim2 = np.random.randint(low=0, high=x_big.shape[1] - dim2)

            # non augmented image
            if non_augmented_array[cpt%n_images]==0:
                if random.Random().random() < (2./nb_augmentations):
                    non_augmented_array[cpt%n_images] = 1
                    x_aug = x_big
                    y_aug = y_big
                    
            patch_x = x_aug[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            patch_y = y_aug[start_dim1:start_dim1 + dim1, start_dim2:start_dim2 + dim2, :]
            
            patch_x = np.asarray(patch_x)
            patch_y = np.asarray(patch_y).astype('int32')

            # save image to buffer
            x[k, :, :, :] = patch_x
            y[k, :, :, :] = patch_y


            cpt += 1
        

        # return the buffer
        yield(x, y)


"""
Training convnets
"""
def weighted_crossentropy(class_weights):

    def func(y_true, y_pred):
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
        weights = tf.reduce_sum(class_weights * y_true, axis=-1)
        weighted_losses = weights * unweighted_losses
        return tf.reduce_mean(weighted_losses)

    return func

def train_model_sample(model = None, dataset_training = None,  dataset_validation = None,
                       model_name = "model", batch_size = 5, n_epoch = 100, 
                       imaging_field_x = 256, imaging_field_y = 256, 
                       output_dir = "./trained_classifiers/", learning_rate = 1e-3, nb_augmentations = 0,
                       train_to_val_ratio = 0.2):

    os.makedirs(name=output_dir, exist_ok=True)
    file_name_save = os.path.join(output_dir, model_name + ".h5")
    logdir = "logs/scalars/" + model_name
    tensorboard_callback = TensorBoard(log_dir=logdir)


    # determine the number of channels and classes
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[-1]
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]

    train_dict, (X_test, Y_test) = get_data_sample(dataset_training, dataset_validation, nb_channels = n_channels, nb_classes = n_classes, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y, nb_augmentations = nb_augmentations, validation_training_ratio = train_to_val_ratio)

    # data information (one way for the user to check if the training dataset makes sense)
    print((nb_augmentations+1)*len(train_dict["channels"]), 'training images')
    print(len(X_test), 'validation images')

    # determine the weights for the weighted cross-entropy based on class distribution for training dataset
    class_weights = np.zeros((n_classes))
    max_number = 0
    class_weights_sum = np.zeros((n_classes))
    for i in range(n_classes):
        for k in range(len(train_dict['labels'])):
            class_weights_sum[i] += np.sum(train_dict['labels'][k][:, :, i])
    for i in range(n_classes):
        if class_weights_sum[i] > max_number:
            max_number = class_weights_sum[i]
    for i in range(n_classes):
        class_weights[i] = max_number / max( class_weights_sum[i], 1.)

    # prepare the model compilation
    optimizer = RMSprop(lr=1e-4)
    model.compile(loss = weighted_crossentropy(class_weights = class_weights), optimizer = optimizer, metrics=['accuracy'])

    # prepare the generation of data
    if nb_augmentations == 0:
        train_generator = random_sample_generator(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_y) 
    else:
        train_generator = random_sample_generator_dataAugmentation(train_dict["channels"], train_dict["labels"], batch_size, n_channels, n_classes, imaging_field_x, imaging_field_y, nb_augmentations) 
        
    # fit the model
    lr_sched = rate_scheduler(lr = learning_rate, decay = 0.95)
    loss_history = model.fit_generator(train_generator,
                                       steps_per_epoch = int((nb_augmentations+1)*len(train_dict["labels"])/batch_size), 
                                       epochs=n_epoch, validation_data=(X_test,Y_test), 
                                       callbacks = [ModelCheckpoint(file_name_save, monitor = 'val_loss', verbose = 0, save_best_only = True, mode = 'auto',save_weights_only = True),LearningRateScheduler(lr_sched),tensorboard_callback])
    

"""
Executing convnets
"""

def get_image_sizes(data_location):
    width = 256
    height = 256
    nb_channels = 1
    img_list = []
    img_list += [getfiles(data_location)]
    img_temp = get_image(os.path.join(data_location, img_list[0][0]))
    if len(img_temp.shape)>2:
        if img_temp.shape[0]<img_temp.shape[2]:
            nb_channels = img_temp.shape[0]
            width = img_temp.shape[1]
            height=img_temp.shape[2]
        else:
            nb_channels = img_temp.shape[2]
            width = img_temp.shape[0]
            height=img_temp.shape[1]
    else:
        width = img_temp.shape[0]
        height=img_temp.shape[1]
    return width, height, nb_channels

def get_images_from_directory(data_location):
    img_list = []
    img_list += [getfiles(data_location)]

    all_images = []
    for stack_iteration in range(len(img_list[0])):
        current_img = get_image(os.path.join(data_location, img_list[0][stack_iteration]))
        all_channels = np.zeros((1, current_img.shape[0], current_img.shape[1], current_img.shape[2]), dtype = 'float32')
        all_channels[0, :, :, :] = current_img
        all_images += [all_channels]
        
    return all_images

def run_model(img, model, imaging_field_x = 256, imaging_field_y = 256):
    
    img[0, :, :, :] = process_image(img[0, :, :, :])
    img = np.pad(img, pad_width = [(0,0), (5,5), (5,5), (0,0)], mode = 'reflect')
            
    n_classes = model.layers[-1].output_shape[-1]
    image_size_x = img.shape[1]
    image_size_y = img.shape[2]
    model_output = np.zeros((image_size_x-10,image_size_y-10,n_classes), dtype = np.float32)
    current_output = np.zeros((1,imaging_field_x,imaging_field_y,n_classes), dtype = np.float32)
    
    x_iterator = 0
    y_iterator = 0
    
    while x_iterator<=(image_size_x-imaging_field_x) and y_iterator<=(image_size_y-imaging_field_y):
        current_output = model.predict(img[:,x_iterator:(x_iterator+imaging_field_x),y_iterator:(y_iterator+imaging_field_y),:])
        model_output[x_iterator:(x_iterator+imaging_field_x-10),y_iterator:(y_iterator+imaging_field_y-10),:] = current_output[:,5:(imaging_field_x-5),5:(imaging_field_y-5),:]
        
        if x_iterator<(image_size_x-2*imaging_field_x):
            x_iterator += (imaging_field_x-10)
        else:
            if x_iterator == (image_size_x-imaging_field_x):
                if y_iterator < (image_size_y-2*imaging_field_y):
                    y_iterator += (imaging_field_y-10)
                    x_iterator = 0
                else:
                    if y_iterator == (image_size_y-imaging_field_y):
                        y_iterator += (imaging_field_y-10)
                    else:
                        y_iterator = (image_size_y-imaging_field_y)
                        x_iterator = 0
            else:
                x_iterator = image_size_x-(imaging_field_x)

    return model_output


def run_models_on_directory(data_location, output_location, model):

    # create output folder if it doesn't exist
    os.makedirs(name=output_location, exist_ok=True)
    
    # determine the number of channels and classes as well as the imaging field dimensions
    input_shape = model.layers[0].output_shape
    n_channels = input_shape[-1]
    imaging_field_x = input_shape[1]
    imaging_field_y = input_shape[2]
    output_shape = model.layers[-1].output_shape
    n_classes = output_shape[-1]
    
    # determine the image size
    image_size_x, image_size_y, nb_channels = get_image_sizes(data_location)
    
    if image_size_x<imaging_field_x:
        sys.exit("The input image has a smaller x dimension than the imaging field")
    if image_size_y<imaging_field_y:
        sys.exit("The input image has a smaller y dimension than the imaging field")
    if n_channels!=nb_channels:
        sys.exit("The input image has a different number of channels than indicated in the U-Net architecture")


    # process images
    counter = 0
    img_list_files = [getfiles(data_location)]

    image_list = get_images_from_directory(data_location)

    for img in image_list:
        print("Processing image ",str(counter + 1)," of ",str(len(image_list)))
        processed_image = run_model(img, model, imaging_field_x = imaging_field_x, imaging_field_y = imaging_field_y)
        
        # Save images
        cnnout_name = os.path.join(output_location, os.path.splitext(img_list_files[0][counter])[0] + ".tiff")
        tiff.imsave(cnnout_name, processed_image)

        counter += 1