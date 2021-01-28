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

from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.optimizers import SGD, RMSprop
    

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