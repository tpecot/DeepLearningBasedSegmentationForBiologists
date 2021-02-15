import mrcnn_config
import mrcnn_utils
import kutils
import skimage.color
import numpy
import os.path
import image_augmentation
import imgaug
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class NucleiConfig(mrcnn_config.Config):
    NAME = "nuclei"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 1 # background + nucleus
    TRAIN_ROIS_PER_IMAGE = 512
    STEPS_PER_EPOCH = 5000 # check mask_train for the final value
    VALIDATION_STEPS = 50
    DETECTION_MAX_INSTANCES = 512
    DETECTION_MIN_CONFIDENCE = 0.5
    DETECTION_NMS_THRESHOLD = 0.35
    RPN_NMS_THRESHOLD = 0.55
    BATCH_SIZE = 1000



class NucleiDataset(mrcnn_utils.Dataset):

    def initialize(self, pImagesAndMasks, pAugmentationLevel = 0):
        self.add_class("nuclei", 1, "nucleus")

        imageIndex = 0

        for imageFile, maskFile in pImagesAndMasks.items():
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            image = skimage.io.imread(imageFile)
            if image.ndim == 1:
                sys.exit("Problem: the images in the training dataset are 1 dimension only")
            if image.ndim == 2:
                image_width = image.shape[1]
                image_height = image.shape[0]
                
            if image.ndim == 3:
                if image.shape[0] < image.shape[2]:
                    image_width = image.shape[2]
                    image_height = image.shape[1]
                else:
                    image_width = image.shape[1]
                    image_height = image.shape[0]
            if image.ndim > 3:
                sys.exit("Problem: the images in the training dataset are more than 2D + C")
            if image.dtype != numpy.uint8:
                image = image.astype('uint8')
                
            self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=image_width, height=image_height, mask_path=maskFile, augmentation_params=None)
            imageIndex += 1

            #adding augmentation parameters
            for augment in range(pAugmentationLevel):
                augmentationMap, widthFactor, heightFactor, switchWidthHeight = image_augmentation.GenerateRandomImgaugAugmentation()
                width = int(float(image.shape[1])*widthFactor)
                height = int(float(image.shape[0])*heightFactor)
                if switchWidthHeight:
                    width, height = height, width
                self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=width, height=height, mask_path=maskFile, augmentation_params=augmentationMap)
                imageIndex += 1


    def initialize_only_augmented(self, pImagesAndMasks, pAugmentationLevel = 0):
        self.add_class("nuclei", 1, "nucleus")

        imageIndex = 0

        for imageFile, maskFile in pImagesAndMasks.items():
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            image = skimage.io.imread(imageFile)
            if image.ndim == 1:
                print("Problem: the images in the training dataset are 1 dimension only")
                continue
            if image.ndim == 2:
                new_image = numpy.zeros((image.shape[0], image.shape[1], 3), numpy.uint8)
                for k in range(3):
                    new_image[:,:,k] = (image[:, :, 0]).astype('uint8')
                image = new_image    
            if image.ndim > 3:
                print("Problem: the images in the training dataset are more than 2D + C")
                continue
            if image.dtype != numpy.uint8:
                image = image.astype('uint8')
                
            #adding augmentation parameters
            for augment in range(pAugmentationLevel):
                augmentationMap, widthFactor, heightFactor, switchWidthHeight = image_augmentation.GenerateRandomImgaugAugmentation()
                width = int(float(image.shape[1])*widthFactor)
                height = int(float(image.shape[0])*heightFactor)
                if switchWidthHeight:
                    width, height = height, width
                self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=width, height=height, mask_path=maskFile, augmentation_params=augmentationMap)
                imageIndex += 1


    def initializePartially(self, pImagesAndMasks, initIndex, endIndex, pAugmentationLevel = 0):
        self.add_class("nuclei", 1, "nucleus")

        imageIndex = 0
        
        for imageFile, maskFile in list(pImagesAndMasks.items())[initIndex:endIndex]:
            baseName = os.path.splitext(os.path.basename(imageFile))[0]

            image = skimage.io.imread(imageFile)
            if image.ndim < 2 or image.dtype != numpy.uint8:
                continue

            self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=image.shape[1], height=image.shape[0], mask_path=maskFile, augmentation_params=None)
            imageIndex += 1

            #adding augmentation parameters
            for augment in range(pAugmentationLevel):
                augmentationMap, widthFactor, heightFactor, switchWidthHeight = image_augmentation.GenerateRandomImgaugAugmentation()
                width = int(float(image.shape[1])*widthFactor)
                height = int(float(image.shape[0])*heightFactor)
                if switchWidthHeight:
                    width, height = height, width
                self.add_image(source="nuclei", image_id=imageIndex, path=imageFile, name=baseName, width=width, height=height, mask_path=maskFile, augmentation_params=augmentationMap)
                imageIndex += 1


    def image_reference(self, image_id):
        info = self.image_info[image_id]
        ref = info["name"]
        augmentation = info["augmentation_params"]

        if augmentation is not None:
            ref + " " + str(augmentation)

        return ref


    def load_image(self, image_id):
        info = self.image_info[image_id]
        imagePath = info["path"]
        augmentation = info["augmentation_params"]

        image = skimage.io.imread(imagePath)
        image = kutils.RCNNConvertInputImage(image)
        
        if augmentation is not None:
            image = augmentation(image = image)

        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        imagePath = info["path"]
        maskPath = info["mask_path"]
        augmentation = info["augmentation_params"]

        image = skimage.io.imread(imagePath)
        image = kutils.RCNNConvertInputImage(image)
        
        mask = skimage.io.imread(maskPath)
        if mask.ndim > 2:
            if mask.shape[0] < mask.shape[2]:
                mask = mask[0, :, :]
            else:
                mask = mask[:,:,0]
        
        
        segmap = SegmentationMapsOnImage(mask, shape=image.shape)
        
        if augmentation is not None:
            segmap = augmentation(segmentation_maps = segmap)
            mask = segmap.get_arr()
        
        maskIndexMax = numpy.max(mask)
        newMaskIndices = numpy.zeros([maskIndexMax], dtype=numpy.uint32)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                index = int(mask[y,x]) - 1
                if index >= 0:
                    newMaskIndices[index] = 1
        count = 0
        for i in range(maskIndexMax):
            if newMaskIndices[i] > 0:
                newMaskIndices[i] = count
                count += 1
        
        masks = numpy.zeros([mask.shape[0], mask.shape[1], count], dtype=numpy.uint8)
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                index = int(mask[y,x]) - 1
                if index >= 0:
                    masks[y,x,newMaskIndices[index]] = 1
                    
        #assign class id 1 to all masks
        class_ids = numpy.array([1 for _ in range(count)])
        return masks, class_ids.astype(numpy.int32)

