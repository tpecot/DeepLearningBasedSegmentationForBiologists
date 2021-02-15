# This program is free software; you can redistribute it and/or modify it under the terms of the GNU Affero General Public License version 3 as published by the Free Software Foundation:
# http://www.gnu.org/licenses/agpl-3.0.txt
############################################################

import sys
import json
import os
import os.path
import numpy
import mrcnn_model
import mrcnn_utils
import mask_rcnn_additional
import random

class MaskTrain:
    __mParams = {}

    def __init__(self, train_dir, eval_dir, input_model, output_dir, model_name, epoch_groups, random_augmentation_level = 100, train_to_val_seed = 0, train_to_val_ratio = 0.0, use_eval_in_val = True, detection_nms_threshold = 0.35, rpn_nms_threshold = 0.65, image_size = 1024):
        self.__mParams["train_dir"] = train_dir
        self.__mParams["eval_dir"] = eval_dir
        self.__mParams["input_model"] = input_model
        self.__mParams["output_dir"] = output_dir
        self.__mParams["model_name"] = model_name
        self.__mParams["epoch_groups"] = epoch_groups
        self.__mParams["random_augmentation_level"] = random_augmentation_level
        self.__mParams["train_to_val_ratio"] = train_to_val_ratio
        self.__mParams["detection_nms_threshold"] = detection_nms_threshold
        self.__mParams["rpn_nms_threshold"] = rpn_nms_threshold
        self.__mParams["image_size"] = image_size
        

    def Train(self):
        fixedRandomSeed = 0
        trainToValidationChance = 0.2
        includeEvaluationInValidation = True
        stepMultiplier = 1.0
        stepCount = 1000
        showInputs = False
        augmentationLevel = 0
        detNMSThresh = 0.35
        rpnNMSThresh = 0.55
        trainDir = os.path.join(os.curdir, self.__mParams["train_dir"])
        evalDir = os.path.join(os.curdir, self.__mParams["eval_dir"])
        inModelPath = os.path.join(os.curdir, self.__mParams["input_model"])
        os.makedirs(name=self.__mParams["output_dir"], exist_ok=True)
        outModelPath = os.path.join(self.__mParams["output_dir"], self.__mParams["model_name"] + ".h5")

        blankInput = True

        if "eval_dir" in self.__mParams:
            evalDir = os.path.join(os.curdir, self.__mParams["eval_dir"])

        if "image_size" in self.__mParams:
            maxdim = self.__mParams["image_size"]

        if "train_to_val_ratio" in self.__mParams:
            trainToValidationChance = self.__mParams["train_to_val_ratio"]

        if "step_num" in self.__mParams:
            stepCount = self.__mParams["step_num"]

        if "show_inputs" in self.__mParams:
            showInputs = self.__mParams["show_inputs"]

        if "random_augmentation_level" in self.__mParams:
            augmentationLevel = self.__mParams["random_augmentation_level"]

        if "detection_nms_threshold" in self.__mParams:
            detNMSThresh = self.__mParams["detection_nms_threshold"]

        if "rpn_nms_threshold" in self.__mParams:
            rpnNMSThresh = self.__mParams["rpn_nms_threshold"]
            

        rnd = random.Random()
        rnd.seed(fixedRandomSeed)
        trainImagesAndMasks = {}
        validationImagesAndMasks = {}

        # iterate through train set
        imagesDir = os.path.join(trainDir, "images")
        masksDir = os.path.join(trainDir, "masks")

        # adding evaluation data into validation
        if includeEvaluationInValidation and evalDir is not None:

            # iterate through test set
            imagesValDir = os.path.join(evalDir, "images")
            masksValDir = os.path.join(evalDir, "masks")

            imageValFileList = [f for f in os.listdir(imagesValDir) if os.path.isfile(os.path.join(imagesValDir, f))]
            for imageFile in imageValFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesValDir, imageFile)
                if os.path.exists(os.path.join(masksValDir, baseName + ".png")):
                    maskPath = os.path.join(masksValDir, baseName + ".png")
                elif os.path.exists(os.path.join(masksValDir, baseName + ".tif")):
                    maskPath = os.path.join(masksValDir, baseName + ".tif")
                elif os.path.exists(os.path.join(masksValDir, baseName + ".tiff")):
                    maskPath = os.path.join(masksValDir, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                validationImagesAndMasks[imagePath] = maskPath
            imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
            for imageFile in imageFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesDir, imageFile)
                if os.path.exists(os.path.join(masksDir, baseName + ".png")):
                    maskPath = os.path.join(masksDir, baseName + ".png")
                elif os.path.exists(os.path.join(masksDir, baseName + ".tif")):
                    maskPath = os.path.join(masksDir, baseName + ".tif")
                elif os.path.exists(os.path.join(masksDir, baseName + ".tiff")):
                    maskPath = os.path.join(masksDir, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                trainImagesAndMasks[imagePath] = maskPath
        # splitting train data into train and validation
        else:
            imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
            for imageFile in imageFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesDir, imageFile)
                if os.path.exists(os.path.join(masksDir, baseName + ".png")):
                    maskPath = os.path.join(masksDir, baseName + ".png")
                elif os.path.exists(os.path.join(masksDir, baseName + ".tif")):
                    maskPath = os.path.join(masksDir, baseName + ".tif")
                elif os.path.exists(os.path.join(masksDir, baseName + ".tiff")):
                    maskPath = os.path.join(masksDir, baseName + ".tiff")
                else:
                    sys.exit("The image " + imageFile + " does not have a corresponding mask file ending with png, tif or tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                if rnd.random() > trainToValidationChance:
                    trainImagesAndMasks[imagePath] = maskPath
                else:
                    validationImagesAndMasks[imagePath] = maskPath

        if len(trainImagesAndMasks) < 1:
            sys.exit("Empty train image list")

        #just to be non-empty
        if len(validationImagesAndMasks) < 1:
            for key, value in trainImagesAndMasks.items():
                validationImagesAndMasks[key] = value
                break

        # Training dataset
        dataset_train = mask_rcnn_additional.NucleiDataset()
        dataset_train.initialize(pImagesAndMasks=trainImagesAndMasks, pAugmentationLevel=augmentationLevel)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = mask_rcnn_additional.NucleiDataset()
        dataset_val.initialize(pImagesAndMasks=validationImagesAndMasks, pAugmentationLevel=0)
        dataset_val.prepare()

        print("training images (with augmentation):", dataset_train.num_images)
        print("validation images (with augmentation):", dataset_val.num_images)

        config = mask_rcnn_additional.NucleiConfig()
        config.IMAGE_MAX_DIM = maxdim
        config.IMAGE_MIN_DIM = maxdim
        config.STEPS_PER_EPOCH = stepCount
        if stepMultiplier is not None:
            steps = int(float(dataset_train.num_images) * stepMultiplier)
            config.STEPS_PER_EPOCH = steps

        config.VALIDATION_STEPS = dataset_val.num_images
        config.DETECTION_NMS_THRESHOLD = detNMSThresh
        config.RPN_NMS_THRESHOLD = rpnNMSThresh
        config.MAX_GT_INSTANCES = 512
        config.BATCH_SIZE = 5000
        config.__init__()
        
        # Create model in training mode
        mdl = mrcnn_model.MaskRCNN(mode="training", config=config, model_dir=os.path.dirname(outModelPath))
        
        if blankInput:
            mdl.load_weights(inModelPath, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            mdl.load_weights(inModelPath, by_name=True)

        allcount = 0
        logdir = "logs/scalars/" + self.__mParams["model_name"]
        for epochgroup in self.__mParams["epoch_groups"]:
            epochs = int(epochgroup["epochs"])
            if epochs < 1:
                continue
            allcount += epochs
            mdl.train(dataset_train,dataset_val,learning_rate=float(epochgroup["learning_rate"]), epochs=allcount,layers=epochgroup["layers"])

        mdl.keras_model.save_weights(outModelPath)
        
        
    def Train_only_augmented(self):
        fixedRandomSeed = None
        trainToValidationChance = 0.2
        includeEvaluationInValidation = True
        stepMultiplier = None
        stepCount = 1000
        showInputs = False
        augmentationLevel = 0
        detNMSThresh = 0.35
        rpnNMSThresh = 0.55
        trainDir = os.path.join(os.curdir, self.__mParams["train_dir"])
        evalDir = os.path.join(os.curdir, self.__mParams["eval_dir"])
        inModelPath = os.path.join(os.curdir, self.__mParams["input_model"])
        outModelPath = os.path.join(os.curdir, self.__mParams["output_model"])
        blankInput = self.__mParams["blank_mrcnn"]

        if "eval_dir" in self.__mParams:
            evalDir = os.path.join(os.curdir, self.__mParams["eval_dir"])

        if "image_size" in self.__mParams:
            maxdim = self.__mParams["image_size"]

        if "train_to_val_seed" in self.__mParams:
            fixedRandomSeed = self.__mParams["train_to_val_seed"]

        if "train_to_val_ratio" in self.__mParams:
            trainToValidationChance = self.__mParams["train_to_val_ratio"]

        if "use_eval_in_val" in self.__mParams:
            includeEvaluationInValidation = self.__mParams["use_eval_in_val"]

        if "step_ratio" in self.__mParams:
            stepMultiplier = self.__mParams["step_ratio"]

        if "step_num" in self.__mParams:
            stepCount = self.__mParams["step_num"]

        if "show_inputs" in self.__mParams:
            showInputs = self.__mParams["show_inputs"]

        if "random_augmentation_level" in self.__mParams:
            augmentationLevel = self.__mParams["random_augmentation_level"]

        if "detection_nms_threshold" in self.__mParams:
            detNMSThresh = self.__mParams["detection_nms_threshold"]

        if "rpn_nms_threshold" in self.__mParams:
            rpnNMSThresh = self.__mParams["rpn_nms_threshold"]
            

        rnd = random.Random()
        rnd.seed(fixedRandomSeed)
        trainImagesAndMasks = {}
        validationImagesAndMasks = {}

        # iterate through train set
        imagesDir = os.path.join(trainDir, "images")
        masksDir = os.path.join(trainDir, "masks")

        # adding evaluation data into validation
        if includeEvaluationInValidation and evalDir is not None:

            # iterate through test set
            imagesValDir = os.path.join(evalDir, "images")
            masksValDir = os.path.join(evalDir, "masks")

            imageValFileList = [f for f in os.listdir(imagesValDir) if os.path.isfile(os.path.join(imagesValDir, f))]
            for imageFile in imageValFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesValDir, imageFile)
                maskPath = os.path.join(masksValDir, baseName + ".tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                validationImagesAndMasks[imagePath] = maskPath
            imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
            for imageFile in imageFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesDir, imageFile)
                maskPath = os.path.join(masksDir, baseName + ".tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                trainImagesAndMasks[imagePath] = maskPath
        # splitting train data into train and validation
        else:
            imageFileList = [f for f in os.listdir(imagesDir) if os.path.isfile(os.path.join(imagesDir, f))]
            for imageFile in imageFileList:
                baseName = os.path.splitext(os.path.basename(imageFile))[0]
                imagePath = os.path.join(imagesDir, imageFile)
                maskPath = os.path.join(masksDir, baseName + ".tiff")
                if not os.path.isfile(imagePath) or not os.path.isfile(maskPath):
                    continue
                if rnd.random() > trainToValidationChance:
                    trainImagesAndMasks[imagePath] = maskPath
                else:
                    validationImagesAndMasks[imagePath] = maskPath

        if len(trainImagesAndMasks) < 1:
            sys.exit("Empty train image list")

        #just to be non-empty
        if len(validationImagesAndMasks) < 1:
            for key, value in trainImagesAndMasks.items():
                validationImagesAndMasks[key] = value
                break

        # Training dataset
        dataset_train = mask_rcnn_additional.NucleiDataset()
        dataset_train.initialize_only_augmented(pImagesAndMasks=trainImagesAndMasks, pAugmentationLevel=augmentationLevel)
        dataset_train.prepare()

        # Validation dataset
        dataset_val = mask_rcnn_additional.NucleiDataset()
        dataset_val.initialize(pImagesAndMasks=validationImagesAndMasks, pAugmentationLevel=0)
        dataset_val.prepare()

        print(dataset_train.num_images, "training images")
        print(dataset_val.num_images, "validation images")

        config = mask_rcnn_additional.NucleiConfig()
        config.IMAGE_MAX_DIM = maxdim
        config.IMAGE_MIN_DIM = maxdim
        config.STEPS_PER_EPOCH = stepCount
        if stepMultiplier is not None:
            steps = int(float(dataset_train.num_images) * stepMultiplier)
            config.STEPS_PER_EPOCH = steps

        config.VALIDATION_STEPS = dataset_val.num_images
        config.DETECTION_NMS_THRESHOLD = detNMSThresh
        config.RPN_NMS_THRESHOLD = rpnNMSThresh
        config.MAX_GT_INSTANCES = 512
        config.BATCH_SIZE = 5000
        config.__init__()
        
        # Create model in training mode
        mdl = mrcnn_model.MaskRCNN(mode="training", config=config, model_dir=os.path.dirname(outModelPath))
        
        if blankInput:
            mdl.load_weights(inModelPath, by_name=True,
                             exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
        else:
            mdl.load_weights(inModelPath, by_name=True)

        allcount = 0
        for epochgroup in self.__mParams["epoch_groups"]:
            epochs = int(epochgroup["epochs"])
            if epochs < 1:
                continue
            allcount += epochs
            mdl.train(dataset_train,dataset_val,learning_rate=float(epochgroup["learning_rate"]), epochs=allcount)

        mdl.keras_model.save_weights(outModelPath)