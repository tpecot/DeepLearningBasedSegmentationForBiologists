# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

import sys
import os
import os.path
import mask_rcnn_additional
import kutils
import numpy
import cv2
import sys
import os
import skimage.morphology
import math
import mrcnn_utils

class Segmentation:
    __mModel = None
    __mConfig = None
    __mModelDir = ""
    __mModelPath = ""
    __mLastMaxDim = mask_rcnn_additional.NucleiConfig().IMAGE_MAX_DIM
    __mConfidence = 0.5
    __NMSThreshold = 0.35

    '''
    @param pModelDir clustering Mask_RCNN model path
    '''
    def __init__(self, pModelPath, pConfidence=0.5, pNMSThreshold = 0.35, pMaxDetNum=512):
        if not os.path.isfile(pModelPath):
            sys.exit("Invalid model path: " + pModelPath)

        self.__mConfidence = pConfidence
        self.__NMSThreshold = pNMSThreshold
        self.__mModelPath = pModelPath
        self.__mModelDir = os.path.dirname(pModelPath)
        self.__mMaxDetNum=pMaxDetNum

    def Segment(self, pImage, pPredictSize=None):

        rebuild = self.__mModel is None

        if pPredictSize is not None:
            maxdim = pPredictSize
            temp = maxdim / 2 ** 6
            if temp != int(temp):
                maxdim = (int(temp) + 1) * 2 ** 6

            if maxdim != self.__mLastMaxDim:
                self.__mLastMaxDim = maxdim
                rebuild = True

        if rebuild:
            import mrcnn_model
            import keras.backend
            keras.backend.clear_session()
            print("Max dim changed (",str(self.__mLastMaxDim),"), rebuilding model")

            self.__mConfig = mask_rcnn_additional.NucleiConfig()
            self.__mConfig.DETECTION_MIN_CONFIDENCE = self.__mConfidence
            self.__mConfig.DETECTION_NMS_THRESHOLD = self.__NMSThreshold
            self.__mConfig.IMAGE_MAX_DIM = self.__mLastMaxDim
            self.__mConfig.IMAGE_MIN_DIM = self.__mLastMaxDim
            self.__mConfig.DETECTION_MAX_INSTANCES=self.__mMaxDetNum
            self.__mConfig.__init__()

            self.__mModel = mrcnn_model.MaskRCNN(mode="inference", config=self.__mConfig, model_dir=self.__mModelDir)
            self.__mModel.load_weights(self.__mModelPath, by_name=True)

        image = kutils.RCNNConvertInputImage(pImage)
        offsetX = 0
        offsetY = 0
        width = image.shape[1]
        height = image.shape[0]

        results = self.__mModel.detect([image], verbose=0)

        r = results[0]
        masks = r['masks']
        scores = r['scores']

        if masks.shape[0] != image.shape[0] or masks.shape[1] != image.shape[1]:
            print("Invalid prediction")
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)


        count = masks.shape[2]
        if count < 1:
            return numpy.zeros((height, width), numpy.uint16), \
                   numpy.zeros((height, width, 0), numpy.uint8),\
                   numpy.zeros(0, numpy.float)

        for i in range(count):
            masks[:, :, i] = numpy.where(masks[:, :, i] == 0, 0, 255)

        return kutils.MergeMasks(masks), masks, scores

    
    def Run(self, imagesDir, outputDir, maxdimValues, subsize_x, subsize_y):
    
        os.makedirs(name=outputDir, exist_ok=True)
        imageFiles = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
        imcount = len(imageFiles)
        for index, imageFile in enumerate(imageFiles):
            print("Image:", str(index + 1), "/", str(imcount), "(", imageFile, ")")

            baseName = os.path.splitext(os.path.basename(imageFile))[0]
            imagePath = os.path.join(imagesDir, imageFile)
            image = skimage.io.imread(imagePath)
            if len(image.shape)>2:
                if image.shape[0]<image.shape[2]:
                    new_image = numpy.zeros((image.shape[1], image.shape[2], 3), numpy.uint16)
                    for k in range(image.shape[0]):
                        new_image[:, :, k] = image[k, :, :]
                    for k in range(image.shape[0], 3):
                        new_image[:, :, k] = image[image.shape[0] - 1, :, :]
                    image = new_image
            elif len(image.shape)==2:
                new_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint16)
                for k in range(3):
                    new_image[:, :, k] = image
                image = new_image
                    
            image_size_x = image.shape[1]
            image_size_y = image.shape[0]
            
            mask_allScales_allImageParts = numpy.zeros((len(maxdimValues),image_size_y,image_size_x), numpy.uint16)
            index = 0
            totalNucleiCount = numpy.zeros(len(maxdimValues))                
            
            for maxdim in maxdimValues:
                x_minIterator = 0
                y_minIterator = 0
                x_maxIterator = subsize_x
                y_maxIterator = subsize_y
                overlap_x = 0
                overlap_y = 0
                if image_size_x<subsize_x:
                    x_maxIterator = image_size_x
                else:
                    overlap_x = math.floor((math.ceil(image_size_x / subsize_x) * subsize_x - image_size_x) / math.floor(image_size_x / subsize_x))
                if image_size_y<subsize_y:
                    y_maxIterator = image_size_y
                else:
                    overlap_y = math.floor((math.ceil(image_size_y / subsize_y) * subsize_y - image_size_y) / math.floor(image_size_y / subsize_y))
                done = False

                while done == False:
                    current_image = image[y_minIterator:y_maxIterator,x_minIterator:x_maxIterator,:]
                
                    mask, masks, scores = self.Segment(pImage=current_image, pPredictSize=maxdim)
                    currentNucleiCount = masks.shape[2]
             
                    x_min_combined = x_minIterator
                    x_range_start = 0
                    if image_size_x<subsize_x:
                        x_range_end = image_size_x
                    else:
                        x_range_end = subsize_x
                    if x_minIterator > 0:
                        x_range_start = math.floor(overlap_x/2)

                    if x_maxIterator != image_size_x:
                        x_range_end = subsize_x - math.ceil(overlap_x/2)

                    y_min_combined = y_minIterator
                    y_range_start = 0
                    if image_size_y<subsize_y:
                        y_range_end = image_size_y
                    else:
                        y_range_end = subsize_y
                    if y_minIterator > 0:
                        y_range_start = math.floor(overlap_y/2)
                    if y_maxIterator != image_size_y:
                        y_range_end = subsize_y - math.ceil(overlap_y/2)

                    nucleiIds = numpy.zeros(numpy.int16(currentNucleiCount)+1, numpy.uint16)

                    if x_min_combined > 0:
                        for y in range(y_range_start, y_range_end):
                            if mask[y,x_range_start] > 0:
                                if mask_allScales_allImageParts[index,y+y_min_combined,x_min_combined+x_range_start-1] > 0:
                                    nucleiIds[mask[y,x_range_start]] = mask_allScales_allImageParts[index,y+y_min_combined,x_min_combined+x_range_start-1]

                    if y_min_combined > 0:
                        for x in range(x_range_start, x_range_end):
                            if mask[y_range_start,x] > 0:
                                if mask_allScales_allImageParts[index,y_min_combined+y_range_start-1,x+x_min_combined] > 0:
                                    nucleiIds[mask[y_range_start,x]] = mask_allScales_allImageParts[index,y_min_combined+y_range_start-1,x+x_min_combined]

                    for y in range(y_range_start, y_range_end):
                        for x in range(x_range_start, x_range_end):
                            if mask[y,x] > 0:
                                if nucleiIds[mask[y,x]] > 0:
                                    mask_allScales_allImageParts[index,y+y_min_combined,x+x_min_combined] = nucleiIds[mask[y,x]]
                                else:
                                    mask_allScales_allImageParts[index,y+y_min_combined,x+x_min_combined] = mask[y,x] + totalNucleiCount[index]

                    totalNucleiCount[index] += currentNucleiCount

                    x_minIterator_memory = x_minIterator
                    x_minIterator = x_maxIterator - overlap_x
                    x_maxIterator = x_minIterator + subsize_x
                    if x_maxIterator > image_size_x:
                        if x_maxIterator < image_size_x + overlap_x:
                            delta_x = x_maxIterator - image_size_x
                            x_minIterator -= delta_x
                            x_maxIterator -= delta_x
                        else:
                            x_minIterator = x_minIterator_memory
                    
                    if x_minIterator_memory==x_minIterator:
                        x_minIterator = 0
                        x_maxIterator = subsize_x

                        y_minIterator_memory = y_minIterator
                        y_minIterator = y_maxIterator - overlap_y
                        y_maxIterator = y_minIterator + subsize_y

                        if y_maxIterator > image_size_y:
                            if y_maxIterator < image_size_y + overlap_y:
                                delta_y = y_maxIterator - image_size_y
                                y_minIterator -= delta_y
                                y_maxIterator -= delta_y
                            else:
                                y_minIterator = y_minIterator_memory
                        
                        if y_minIterator_memory==y_minIterator:
                            done = True

                index = index+1

            skimage.io.imsave(os.path.join(outputDir, baseName + ".tiff"), mask_allScales_allImageParts)
