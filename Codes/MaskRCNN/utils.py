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

import ipywidgets as widgets
import ipyfilechooser
from ipyfilechooser import FileChooser
from ipywidgets import HBox, Label, Layout

import sys
sys.path.append("Mask_RCNN-2.1")
import mrcnn_model
import mrcnn_utils
sys.path.append("biomagdsb")
import mask_rcnn_additional
import additional_train
import additional_segmentation


"""
Interfaces
"""

def TensorBoard_interface():
    
    print('\x1b[1m'+"Model location")
    classifier_directory = FileChooser('./trainedClassifiers')
    display(classifier_directory)
    
    return classifier_directory

def extract_channels_interface():
    
    parameters = []
    
    print('\x1b[1m'+"Input directory")
    input_dir = FileChooser('./datasets')
    display(input_dir)
    print('\x1b[1m'+"Output directory")
    output_dir = FileChooser('./datasets')
    display(output_dir)
    channel_1 = widgets.Checkbox(value=True, description='Channel 1',disabled=False)
    display(channel_1)
    channel_2 = widgets.Checkbox(value=False, description='Channel 2',disabled=False)
    display(channel_2)
    channel_3 = widgets.Checkbox(value=False, description='Channel 3',disabled=False)
    display(channel_3)
    channel_4 = widgets.Checkbox(value=False, description='Channel 4',disabled=False)
    display(channel_4)
    channel_5 = widgets.Checkbox(value=False, description='Channel 5',disabled=False)
    display(channel_5)
    channel_6 = widgets.Checkbox(value=False, description='Channel 6',disabled=False)
    display(channel_6)
    channel_7 = widgets.Checkbox(value=False, description='Channel 7',disabled=False)
    display(channel_7)

    parameters.append(input_dir)
    parameters.append(output_dir)
    parameters.append(channel_1)
    parameters.append(channel_2)
    parameters.append(channel_3)
    parameters.append(channel_4)
    parameters.append(channel_5)
    parameters.append(channel_6)
    parameters.append(channel_7)
    return parameters

def divide_images_interface():
    
    parameters = []
    
    print('\x1b[1m'+"Input directory")
    input_dir = FileChooser('./datasets')
    display(input_dir)
    print('\x1b[1m'+"Output directory")
    output_dir = FileChooser('./datasets')
    display(output_dir)

    label_layout = Layout(width='100px',height='30px')

    width_divider = HBox([Label('Width divider:', layout=label_layout), widgets.IntText(
        value=2, description='', disabled=False)])
    display(width_divider)
    height_divider = HBox([Label('Height divider:', layout=label_layout), widgets.IntText(
        value=2, description='', disabled=False)])
    display(height_divider)
    
    parameters.append(input_dir)
    parameters.append(output_dir)
    parameters.append(width_divider)
    parameters.append(height_divider)
    return parameters

def training_parameters_interface(nb_trainings):
    training_dir = np.zeros([nb_trainings], FileChooser)
    validation_dir = np.zeros([nb_trainings], FileChooser)
    input_model = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    heads_training = np.zeros([nb_trainings], HBox)
    nb_epochs_heads = np.zeros([nb_trainings], HBox)
    learning_rate_heads = np.zeros([nb_trainings], HBox)
    all_network_training = np.zeros([nb_trainings], HBox)
    nb_epochs_all = np.zeros([nb_trainings], HBox)
    learning_rate_all = np.zeros([nb_trainings], HBox)
    nb_augmentations = np.zeros([nb_trainings], HBox)
    train_to_val_ratio = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Training directory")
        training_dir[i] = FileChooser('./datasets')
        display(training_dir[i])
        print('\x1b[1m'+"Validation directory")
        validation_dir[i] = FileChooser('./datasets')
        display(validation_dir[i])
        print('\x1b[1m'+"Input model")
        input_model[i] = FileChooser('./pretrainedClassifiers')
        display(input_model[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./trainedClassifiers')
        display(output_dir[i])

        label_layout = Layout(width='250px',height='30px')

        heads_training[i] = HBox([Label('Training heads only first:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(heads_training[i])

        nb_epochs_heads[i] = HBox([Label('Number of epochs for heads training:', layout=label_layout), widgets.IntText(
            value=1, description='', disabled=False)])
        display(nb_epochs_heads[i])

        learning_rate_heads[i] = HBox([Label('Learning rate for heads training:', layout=label_layout), widgets.FloatText(
            value=0.001, description='', disabled=False)])
        display(learning_rate_heads[i])

        all_network_training[i] = HBox([Label('Training all network:', layout=label_layout), widgets.Checkbox(
            value=True, description='',disabled=False)])
        display(all_network_training[i])

        nb_epochs_all[i] = HBox([Label('Number of epochs for all network training:', layout=label_layout), widgets.IntText(
            value=3, description='', disabled=False)])
        display(nb_epochs_all[i])

        learning_rate_all[i] = HBox([Label('Learning rate for all network training:', layout=label_layout), widgets.FloatText(
            value=0.0005, description='', disabled=False)])
        display(learning_rate_all[i])

        nb_augmentations[i] = HBox([Label('Number of augmentations:', layout=label_layout), widgets.IntText(
            value=100, description='', disabled=False)])
        display(nb_augmentations[i])

        train_to_val_ratio[i] = HBox([Label('Ratio of training in validation:', layout=label_layout), widgets.BoundedFloatText(
            value=0.2, min=0.01, max=0.99, step=0.01, description='', disabled=False, color='black'
        )])
        display(train_to_val_ratio[i])

    parameters.append(training_dir)
    parameters.append(validation_dir)
    parameters.append(input_model)
    parameters.append(output_dir)
    parameters.append(heads_training)
    parameters.append(nb_epochs_heads)
    parameters.append(learning_rate_heads)
    parameters.append(all_network_training)
    parameters.append(nb_epochs_all)
    parameters.append(learning_rate_all)
    parameters.append(nb_augmentations)
    parameters.append(train_to_val_ratio)
    
    return parameters  

def running_parameters_interface(nb_trainings):
    input_dir = np.zeros([nb_trainings], FileChooser)
    input_classifier = np.zeros([nb_trainings], FileChooser)
    output_dir = np.zeros([nb_trainings], FileChooser)
    image_size = np.zeros([nb_trainings], HBox)
    
    parameters = []
    for i in range(nb_trainings):
        print('\x1b[1m'+"Input directory")
        input_dir[i] = FileChooser('./datasets')
        display(input_dir[i])
        print('\x1b[1m'+"Input classifier")
        input_classifier[i] = FileChooser('./trainedClassifiers')
        display(input_classifier[i])
        print('\x1b[1m'+"Output directory")
        output_dir[i] = FileChooser('./datasets')
        display(output_dir[i])

        label_layout = Layout(width='215px',height='30px')

        image_size[i] = HBox([Label('Image size as seen by the network:', layout=label_layout), widgets.IntText(
            value=1024, description='', disabled=False)])
        display(image_size[i])

    parameters.append(input_dir)
    parameters.append(input_classifier)
    parameters.append(output_dir)
    parameters.append(image_size)
    
    return parameters  

def combine_instance_segmentations_interface():
    
    parameters = []
    
    print('\x1b[1m'+"Input directory for Mask R-CNN segmentations")
    input_dir1 = FileChooser('./datasets')
    display(input_dir1)
    print('\x1b[1m'+"Input directory for U-Net segmentations")
    input_dir2 = FileChooser('./datasets')
    display(input_dir2)
    print('\x1b[1m'+"Output directory")
    output_dir = FileChooser('./datasets')
    display(output_dir)

    parameters.append(input_dir1)
    parameters.append(input_dir2)
    parameters.append(output_dir)
    return parameters


"""
Pre-processing functions 
"""

def extract_channels(parameters):
    if parameters[0].selected==None:
        sys.exit("You need to select an input directory")
    if parameters[1].selected==None:
        sys.exit("You need to select an output directory")
    imageFiles = [f for f in os.listdir(parameters[0].selected) if os.path.isfile(os.path.join(parameters[0].selected, f))]
    os.makedirs(name=parameters[1].selected, exist_ok=True)

    channels = []
    if parameters[2].value==True:
        channels.append(0)
    if parameters[3].value==True:
        channels.append(1)
    if parameters[4].value==True:
        channels.append(2)
    if parameters[5].value==True:
        channels.append(3)
    if parameters[6].value==True:
        channels.append(4)
    if parameters[7].value==True:
        channels.append(5)
    if parameters[8].value==True:
        channels.append(6)
    if len(channels)==0:
        sys.exit("You need to select at least one channel")
    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(parameters[0].selected, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = skimage.io.imread(imagePath)
        if image.shape[0]<image.shape[-1]:
            output_image = np.zeros((len(channels), image.shape[1], image.shape[2]), np.uint16)
            for i in range(len(channels)):
                output_image[i, :, :] = (image[channels[i], :, :]).astype('uint16')
            tiff.imsave(os.path.join(parameters[1].selected, baseName + ".tiff"), output_image)
            
        else:
            output_image = np.zeros((len(channels), image.shape[0], image.shape[1]), np.uint16)
            for i in range(len(channels)):
                output_image[i, :, :] = (image[:, :,channels[i]]).astype('uint16')
            tiff.imsave(os.path.join(parameters[1].selected, baseName + ".tiff"), output_image)

def divide_images(parameters):
    if parameters[0].selected==None:
        sys.exit("You need to select an input directory")
    if parameters[1].selected==None:
        sys.exit("You need to select an output directory")
    imageFiles = [f for f in os.listdir(parameters[0].selected) if os.path.isfile(os.path.join(parameters[0].selected, f))]
    os.makedirs(name=parameters[1].selected, exist_ok=True)

    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(parameters[0].selected, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = skimage.io.imread(imagePath)
        
        width_channel = 0
        height_channel = 1
        nb_channels = 1
        if len(image.shape)>2:
            if image.shape[0]<image.shape[-1]:
                width = int(image.shape[1]/parameters[3].children[1].value)
                height = int(image.shape[2]/parameters[2].children[1].value)
                width_channel = 1
                height_channel = 2
                nb_Channels = image.shape[0]
            else:
                width = int(image.shape[0]/parameters[3].children[1].value)
                height = int(image.shape[1]/parameters[2].children[1].value)
                nb_Channels = image.shape[2]
        else:
            width = int(image.shape[0]/parameters[3].children[1].value)
            height = int(image.shape[1]/parameters[2].children[1].value)
                
        for i in range(parameters[3].children[1].value):
            for j in range(parameters[2].children[1].value):
                x_init = int((image.shape[width_channel]/parameters[3].children[1].value)*i)
                x_end = x_init + width
                y_init = int((image.shape[height_channel]/parameters[2].children[1].value)*j)
                y_end = y_init + height
            
                if len(image.shape)==2:
                    output_image = np.zeros((width, height), np.uint16)
                    output_image = (image[x_init:x_end, y_init:y_end]).astype('uint16')
                    tiff.imsave(os.path.join(parameters[1].selected, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                else:
                    output_image = np.zeros((nb_channels, width, height), np.uint16)
                    if image.shape[0]<image.shape[-1]:
                        output_image = (image[:, x_init:x_end, y_init:y_end]).astype('uint16')
                        tiff.imsave(os.path.join(parameters[1].selected, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)
                    else:
                        output_image = (image[x_init:x_end, y_init:y_end, :]).astype('uint16')
                        tiff.imsave(os.path.join(parameters[1].selected, baseName + "_" + str(i) + "_" + str(j) + ".tiff"), output_image)



"""
Training and processing calling functions 
"""

def training(nb_trainings, parameters):
    for i in range(nb_trainings):
        if parameters[0][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an input directory for training")
        if parameters[2][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an inmput model for transfer learning")
        if parameters[3][i].selected==None:
            sys.exit("Training #"+str(i+1)+": You need to select an output directory for the trained classifier")
    
        model_name = "MaskRCNN_"+str(parameters[6][i].children[1].value)+"_lr_heads_"+str(parameters[5][i].children[1].value)+"ep_heads_"+str(parameters[9][i].children[1].value)+"_lr_all_"+str(parameters[8][i].children[1].value)+"ep_all_"+str(parameters[10][i].children[1].value)+"DA"

        if parameters[4][i].children[1].value==True and parameters[7][i].children[1].value==True:
            epoch_groups = [{"layers":"heads","epochs":str(parameters[5][i].children[1].value),"learning_rate":str(parameters[6][i].children[1].value)}, {"layers":"all","epochs":str(parameters[8][i].children[1].value),"learning_rate":str(parameters[9][i].children[1].value)}]
        else:
            if parameters[4][i].children[1].value==True:
                epoch_groups = [{"layers":"heads","epochs":str(parameters[5][i].children[1].value),"learning_rate":str(parameters[6][i].children[1].value)}]
            elif parameters[7][i].children[1].value==True:
                epoch_groups = [{"layers":"all","epochs":str(parameters[8][i].children[1].value),"learning_rate":str(parameters[9][i].children[1].value)}]
            else:
                sys.exit("Training #"+str(i+1)+": You need to train heads, all network or both")

        model = additional_train.MaskTrain(parameters[0][i].selected, parameters[1][i].selected, parameters[2][i].selected, parameters[3][i].selected, model_name, epoch_groups, parameters[10][i].children[1].value, 0, parameters[11][i].children[1].value, True, 0.5, 0.6, 512)
        model.Train()
        
        

def running(nb_runnings, parameters):
    for i in range(nb_runnings):
        if parameters[0][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an input directory for images to be processed")
        if parameters[1][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select a trained model to run your images")
        if parameters[2][i].selected==None:
            sys.exit("Running #"+str(i+1)+": You need to select an output directory for processed images")

        model = additional_segmentation.Segmentation(parameters[1][i].selected, 0.5, 0.35, 2000)
        model.Run(parameters[0][i].selected, parameters[2][i].selected, [parameters[3][i].children[1].value], 512, 512)
        del model
    

"""
Post-processing functions 
"""
def combine_instance_segmentations(parameters):
    if parameters[0].selected==None:
        sys.exit("You need to select an input directory for the Mask R-CNN segmentations")
    if parameters[1].selected==None:
        sys.exit("You need to select an input directory for the U-Net segmentations")
    if parameters[2].selected==None:
        sys.exit("You need to select an output directory")
    imageFiles1 = [f for f in os.listdir(parameters[0].selected) if os.path.isfile(os.path.join(parameters[0].selected, f))]
    os.makedirs(name=parameters[2].selected, exist_ok=True)
    
    for index, imageFile in enumerate(imageFiles1):
        imagePath1 = os.path.join(parameters[0].selected, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image1 = skimage.io.imread(imagePath1)
        imagePath2 = os.path.join(parameters[1].selected, imageFile)
        image2 = skimage.io.imread(imagePath2)

        if len(image1.shape)>2:
            if image1.shape[0]<image1.shape[2]:
                new_image1 = np.zeros((image1.shape[1], image1.shape[2]), np.uint16)
                new_image1[:, :] = image1[0, :, :]
            else:
                new_image1 = np.zeros((image1.shape[0], image1.shape[1]), np.uint16)
                new_image1[:, :] = image1[:, :, 0]
            image1 = new_image1
        if len(image2.shape)>2:
            if image2.shape[0]<image2.shape[2]:
                new_image2 = np.zeros((image2.shape[1], image2.shape[2]), np.uint16)
                new_image2[:, :] = image2[0, :, :]
            else:
                new_image2 = np.zeros((image2.shape[0], image2.shape[1]), np.uint16)
                new_image2[:, :] = image2[:, :, 0]
            image2 = new_image2
            
        image1IndexMax = np.max(image1)
        image1Indices = np.zeros([int(image1IndexMax)], dtype=np.uint32)
        image2IndexMax = np.max(image2)
        image2Indices = np.zeros([int(image2IndexMax), 3], dtype=np.uint32)
        output = np.zeros([image1.shape[0], image1.shape[1]], dtype=np.uint32)
        for y in range(image2.shape[0]):
            for x in range(image2.shape[1]):
                index1 = int(image1[y,x]) - 1
                index2 = int(image2[y,x]) - 1
                if index1 >= 0:
                    if index2 >= 0:
                        image1Indices[index1] = 1
                if index2 >= 0:
                    image2Indices[index2, 1] = image2Indices[index2, 1] + 1
                    if index1 >= 0:
                        image2Indices[index2, 0] = image2Indices[index2, 0] + 1
                
        currentNucleusIndex = 1
        for i in range(int(image1IndexMax)):
            if image1Indices[i] > 0:
                image1Indices[i] = currentNucleusIndex
                currentNucleusIndex = currentNucleusIndex + 1
                
        for y in range(image2.shape[0]):
            for x in range(image2.shape[1]):
                index = int(image1[y,x]) - 1
                if index >= 0:
                    if image1Indices[index] > 0:
                        output[y,x] = image1Indices[index]
        
        for i in range(int(image2IndexMax)):
            if image2Indices[i, 1]>0:
                if image2Indices[i, 0]>0:
                    if float(image2Indices[i, 1])/float(image2Indices[i, 0])>2.:
                        image2Indices[i, 2] = currentNucleusIndex
                        currentNucleusIndex = currentNucleusIndex + 1
                else:
                    image2Indices[i, 2] = currentNucleusIndex
                    currentNucleusIndex = currentNucleusIndex + 1

        for y in range(image2.shape[0]):
            for x in range(image2.shape[1]):
                index = int(image2[y,x]) - 1
                if index >= 0:
                    if image2Indices[index, 2] > 0:
                        output[y,x] = image2Indices[index, 2]
            
        tiff.imsave(os.path.join(parameters[2].selected, baseName + ".tiff"), output)
 

## functions needed to merge scales
def intersections(nucleus1, nucleus2):
    overlap = 0
    for i in range(len(nucleus1[0])):
        for j in range(len(nucleus2[0])):
            if ( (nucleus2[0][j] == nucleus1[0][i]) and (nucleus2[1][j] == nucleus1[1][i]) ):
                overlap = overlap + 1
    return ( overlap / (len(nucleus1[0])) ), ( overlap / (len(nucleus2[0])) )


def merge_nuclei(scales, indices):
    # create new nucleus
    x_coords = nucleiPerScale[scales[0]][currentNucleiToModify[scales[0]][indices[0]]][0]
    y_coords = nucleiPerScale[scales[0]][currentNucleiToModify[scales[0]][indices[0]]][1]
    for i in range(len(scales)):
        if i > 0:
            x_coords = np.append(x_coords, nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0])
            y_coords = np.append(y_coords, nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1])
    newNucleus = list((x_coords, y_coords))

    # add nucleus 
    nucleiPerScale[scales[0]].append(newNucleus)
    # update id
    for p in range(len(nucleiToModify_initialIds_perScale[scales[0]])):
        #print(nucleiToModify_initialIds_perScale[scales[0]][p], " ", currentNucleiToModify[scales[0]][indices[0]])
        if nucleiToModify_initialIds_perScale[scales[0]][p] == currentNucleiToModify[scales[0]][indices[0]]:
            nucleiToModify_initialIds_perScale[scales[0]][p] = len(nucleiPerScale[scales[0]]) - 1 
            currentNucleiToModify[scales[0]][indices[0]] = len(nucleiPerScale[scales[0]]) - 1
    # remove other nucleus-i
    for i in range(len(scales)):
        if i > 0:
            for c in range(image.shape[0]):
                indexToRemove = -1
                for p in range(len(nucleiToModify_initialIds_perScale[c])):
                    if nucleiToModify_initialIds_perScale[c][p] == currentNucleiToModify[scales[i]][indices[i]]:
                        indexToRemove = p
                if indexToRemove > -1:
                    nucleiToModify_initialIds_perScale[c].pop(indexToRemove)

def merge_nuclei_and_add(scales, indices):
    # create new nucleus
    x_coords = nucleiPerScale[scales[0]][currentNucleiToModify[scales[0]][indices[0]]][0]
    y_coords = nucleiPerScale[scales[0]][currentNucleiToModify[scales[0]][indices[0]]][1]
    for i in range(len(scales)):
        if i > 0:
            x_coords = np.append(x_coords, nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0])
            y_coords = np.append(y_coords, nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1])
    newNucleus = list((x_coords, y_coords))

    # add nucleus as final nucleus
    finalNuclei.append(newNucleus)
    
def split_nuclei(scales, indices):
    nb_nuclei = len(scales)
        
    nuclei_centers = []
    new_nuclei = []
    for n in range(nb_nuclei):
        empty_table_1 = []
        empty_table_2 = []
        empty_coord = []
        empty_coord.append(empty_table_1)
        empty_coord.append(empty_table_2)
        nuclei_centers.append(empty_coord)
        empty_table_x = []
        empty_table_y = []
        double_empty_table_x = []
        double_empty_table_y = []
        double_empty_table_x.append(empty_table_x)
        double_empty_table_y.append(empty_table_y)
        double_empty_coord = []
        double_empty_coord.append(double_empty_table_x)
        double_empty_coord.append(double_empty_table_y)
        new_nuclei.append(double_empty_coord)

    current_nucleus = 0
    for i in range(len(scales)):
        nuclei_centers[current_nucleus][0] = np.average(nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0])
        nuclei_centers[current_nucleus][1] = np.average(nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1])
        current_nucleus += 1
        
    already_defined_nuclei = np.zeros(nb_nuclei)
    for i in range(len(scales)):
        for p in range(len(nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0])):
            min_distance = 100000
            index_ref = 0
            for n in range(len(nuclei_centers)):
                if ( (nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0][p] - nuclei_centers[n][0])**2 + (nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1][p] - nuclei_centers[n][1])**2 ) < min_distance :
                    min_distance = (nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0][p] - nuclei_centers[n][0])**2 + (nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1][p] - nuclei_centers[n][1])**2
                    index_ref = n
            if already_defined_nuclei[index_ref] < 1:
                new_nuclei[index_ref][0][0] = nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0][p]
                new_nuclei[index_ref][1][0] = nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1][p]
                already_defined_nuclei[index_ref] = 10
            else:
                new_nuclei[index_ref][0].append(nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][0][p])
                new_nuclei[index_ref][1].append(nucleiPerScale[scales[i]][currentNucleiToModify[scales[i]][indices[i]]][1][p])
        
        ##############################################################################################
    # add nucleus
    for i in range(len(scales)):
        nucleiPerScale[scales[i]].append(list((new_nuclei[i][0], new_nuclei[i][1])))
        for p in range(len(nucleiToModify_initialIds_perScale[scales[i]])):
            if nucleiToModify_initialIds_perScale[scales[i]][p] == currentNucleiToModify[scales[i]][indices[i]]:
                nucleiToModify_initialIds_perScale[scales[i]][p] = len(nucleiPerScale[scales[i]]) - 1 
                currentNucleiToModify[scales[i]][indices[i]] = len(nucleiPerScale[scales[i]]) - 1
                
def clean_all():
    # cleaning
    for c in range(image.shape[0]):
        indicesToRemove = []
        for i in range(len(currentNucleiToModify[c])):
            for j in range(len(nucleiToModify_initialIds_perScale[c])):
                if nucleiToModify_initialIds_perScale[c][j] == currentNucleiToModify[c][i]:
                    indicesToRemove.append(j)
        indicesToRemove = np.sort(indicesToRemove)            
        for i in range(len(indicesToRemove)):
            nucleiToModify_initialIds_perScale[c].pop(indicesToRemove[i]-i)

def merge_scales(input_image_dir, output_image_dir):
    imageFiles = [f for f in os.listdir(input_image_dir) if os.path.isfile(os.path.join(input_image_dir, f))]
    os.makedirs(name=output_image_dir, exist_ok=True)

    for index, imageFile in enumerate(imageFiles):
        imagePath = os.path.join(input_image_dir, imageFile)
        baseName = os.path.splitext(os.path.basename(imageFile))[0]
        image = skimage.io.imread(imagePath)
    
        nucleiPerScale = []
        for c in range(image.shape[0]):
            nucleiForCurrentScale = []
            for i in range(np.max(image[c,:,:])+1):
                if i > 0:
                    indices = np.where(image[c,:,:] == i)
                    nucleiForCurrentScale.append(indices)
            nucleiPerScale.append(nucleiForCurrentScale)
    
        finalNuclei = []
        nucleiToModify_initialIds_perScale = []
        for c in range(image.shape[0]):
            nucleiToModify_initialIds = []
            for i in range(len(nucleiPerScale[c])):
                newNucleus = True
                for d in range(image.shape[0]):
                    if (d != c):
                        for p in range(len(nucleiPerScale[c][i][0])):
                            if image[d, nucleiPerScale[c][i][0][p], nucleiPerScale[c][i][1][p]] > 0:
                                newNucleus = False
                if newNucleus:
                    finalNuclei.append(nucleiPerScale[c][i])
                else:
                    nucleiToModify_initialIds.append(i)
            nucleiToModify_initialIds_perScale.append(nucleiToModify_initialIds)

        currentNucleiToModify = []
        for nuc in range(image.shape[0]):
            while len(nucleiToModify_initialIds_perScale[nuc]) > 0:
                currentNucleiToModify = []
                currentNucleiToModify_perScale = []
                currentNucleiToModify_perScale.append(nucleiToModify_initialIds_perScale[nuc][0])
                for c in range(image.shape[0]):
                    if c != nuc:
                        emptyNucleus = []
                        currentNucleiToModify.append(emptyNucleus)
                    else:
                        currentNucleiToModify.append(currentNucleiToModify_perScale)
                                        
        
                nbElements = 1
                nbElementsMem = 0
                while nbElements != nbElementsMem:
                    nbElementsMem = nbElements
                    for c in range(image.shape[0]):
                        for i in range(len(currentNucleiToModify[c])):
                            for d in range(image.shape[0]):
                                if (d != c):
                                    for p in range(len(nucleiPerScale[c][currentNucleiToModify[c][i]][0])):
                                        if image[d, nucleiPerScale[c][currentNucleiToModify[c][i]][0][p], nucleiPerScale[c][currentNucleiToModify[c][i]][1][p]] > 0:
                                            nucleusIndex = image[d, nucleiPerScale[c][currentNucleiToModify[c][i]][0][p], nucleiPerScale[c][currentNucleiToModify[c][i]][1][p]]-1
                                            alreadyRegistered = False
                                            for k in range(len(currentNucleiToModify[d])):
                                                if currentNucleiToModify[d][k] == nucleusIndex:
                                                    alreadyRegistered = True
                                            if alreadyRegistered == False:
                                                currentNucleiToModify[d].append(nucleusIndex)
                    nbElements = 0
                    for c in range(image.shape[0]):
                        nbElements += len(currentNucleiToModify[c])
        
                nb_nuclei_perScale = np.zeros(image.shape[0], numpy.uint8)
                for c in range(image.shape[0]):
                    nb_nuclei_perScale[c] = len(currentNucleiToModify[c])
        
                most_common_nuclei_number = np.zeros(int(np.max(nb_nuclei_perScale))+1, numpy.uint8)
                for c in range(image.shape[0]):
                    most_common_nuclei_number[nb_nuclei_perScale[c]] += 1
        
                max_number_votes = 0
                max_number_votes_index = 1
                for i in range(most_common_nuclei_number.shape[0]):
                    if i > 0:
                        if most_common_nuclei_number[i] > max_number_votes:
                            max_number_votes = most_common_nuclei_number[i]
                            max_number_votes_index = i
        
                if max_number_votes_index < 2:
                    scales = []
                    indices = []
                    for c in range(image.shape[0]):
                        for i in range(len(currentNucleiToModify[c])):
                            scales.append(c)
                            indices.append(i)
                    merge_nuclei_and_add(scales, indices)
                    clean_all()
            
                else:
                    nuclei_to_consider = np.zeros(image.shape[0], numpy.uint8)
                    min_index = -1
                    for c in range(image.shape[0]):
                        if nb_nuclei_perScale[c] == max_number_votes_index:
                            nuclei_to_consider[c] = 1
                            if min_index < 0:
                                min_index = c
                        
                    for i in range(len(currentNucleiToModify[min_index])):
                        scales = []
                        indices = []
                        scales.append(min_index)
                        indices.append(i)
                        for c in range(min_index+1, image.shape[0]):
                            if nuclei_to_consider[c] > 0:
                                for j in range(len(currentNucleiToModify[c])):
                            
                                    score1, score2 = intersections(nucleiPerScale[min_index][currentNucleiToModify[min_index][i]], nucleiPerScale[c][currentNucleiToModify[c][j]])
                                    if score1 > 0.5 or score2 > 0.5:
                                        scales.append(c)
                                        indices.append(j)
                        if len(scales) > 1:
                            merge_nuclei(scales, indices)
                            
                    scales = []
                    indices = []
                    for i in range(len(currentNucleiToModify[min_index])):
                        scales.append(min_index)
                        indices.append(i)
                    split_nuclei(scales, indices)
                    new_nuclei = []
                    for n in range(len(currentNucleiToModify[min_index])):
                        empty_table_1 = []
                        empty_table_2 = []
                        empty_coord = []
                        empty_coord.append(empty_table_1)
                        empty_coord.append(empty_table_2)
                        empty_table_x = []
                        empty_table_y = []
                        double_empty_table_x = []
                        double_empty_table_y = []
                        double_empty_table_x.append(empty_table_x)
                        double_empty_table_y.append(empty_table_y)
                        double_empty_coord = []
                        double_empty_coord.append(double_empty_table_x)
                        double_empty_coord.append(double_empty_table_y)
                        new_nuclei.append(double_empty_coord)

                    nuclei_centers_x = np.zeros(max_number_votes_index)
                    nuclei_centers_y = np.zeros(max_number_votes_index)
                    already_defined_nuclei = np.zeros(len(currentNucleiToModify[min_index]))
                    for i in range(len(currentNucleiToModify[min_index])):
                        nuclei_centers_x[i] = np.average(nucleiPerScale[min_index][currentNucleiToModify[min_index][i]][0])
                        nuclei_centers_y[i] = np.average(nucleiPerScale[min_index][currentNucleiToModify[min_index][i]][1])
                        for p in range(len(nucleiPerScale[min_index][currentNucleiToModify[min_index][i]][0])):
                            if already_defined_nuclei[i] < 1:
                                new_nuclei[i][0][0] = nucleiPerScale[scales[i]][currentNucleiToModify[min_index][i]][0][p]
                                new_nuclei[i][1][0] = nucleiPerScale[scales[i]][currentNucleiToModify[min_index][i]][1][p]
                                already_defined_nuclei[i] = 10
                            else:
                                new_nuclei[i][0].append(nucleiPerScale[scales[i]][currentNucleiToModify[min_index][i]][0][p])
                                new_nuclei[i][1].append(nucleiPerScale[scales[i]][currentNucleiToModify[min_index][i]][1][p])
        

                    for c in range(image.shape[0]):
                        if c != min_index:
                            for i in range(len(currentNucleiToModify[c])):
                                for p in range(len(nucleiPerScale[c][currentNucleiToModify[c][i]][0])):
                                    min_distance = 100000
                                    index_ref = 0
                                    for n in range(len(currentNucleiToModify[min_index])):
                                        if ( (nucleiPerScale[c][currentNucleiToModify[c][i]][0][p] - nuclei_centers_x[n])**2 + (nucleiPerScale[c][currentNucleiToModify[c][i]][1][p] - nuclei_centers_y[n])**2 ) < min_distance :
                                            min_distance = (nucleiPerScale[c][currentNucleiToModify[c][i]][0][p] - nuclei_centers_x[n])**2 + (nucleiPerScale[c][currentNucleiToModify[c][i]][1][p] - nuclei_centers_y[n])**2 
                                            index_ref = n
                                    new_nuclei[index_ref][0].append(nucleiPerScale[c][currentNucleiToModify[c][i]][0][p])
                                    new_nuclei[index_ref][1].append(nucleiPerScale[c][currentNucleiToModify[c][i]][1][p])
        
                    # add nuclei as final nucleus
                    for i in range(len(currentNucleiToModify[min_index])):
                        finalNuclei.append(new_nuclei[i])

                    clean_all()

        output = numpy.zeros((image.shape[1], image.shape[2]), numpy.uint32)
        for n in range(len(finalNuclei)):
            for i in range(len(finalNuclei[n][0])):
                output[finalNuclei[n][0][i], finalNuclei[n][1][i]] = n+1
        tiff.imsave(os.path.join(output_image_dir, baseName + ".tiff"), output)