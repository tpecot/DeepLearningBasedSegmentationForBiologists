import numpy
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
import random
import copy
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import skimage.io



def GenerateRandomImgaugAugmentation(
        pNbAugmentations=5,           # number of augmentations
        pEnableResizing=True,          # enable scaling
        pScaleFactor=0.5,              # maximum scale factor
        pEnableCropping=True,           # enable cropping
        pCropFactor=0.25,               # maximum crop out size (minimum new size is 1.0-pCropFactor)
        pEnableFlipping1=True,          # enable x flipping
        pEnableFlipping2=True,          # enable y flipping
        pEnableRotation90=True,           # enable rotation
        pEnableRotation=True,           # enable rotation
        pMaxRotationDegree=15,             # maximum shear degree
        pEnableShearX=True,             # enable x shear
        pEnableShearY=True,             # enable y shear
        pMaxShearDegree=15,             # maximum shear degree
        pEnableDropOut=True,            # enable pixel dropout
        pMaxDropoutPercentage=.1,     # maximum dropout percentage
        pEnableBlur=True,               # enable gaussian blur
        pBlurSigma=.25,                  # maximum sigma for gaussian blur
        pEnableSharpness=True,          # enable sharpness
        pSharpnessFactor=.1,           # maximum additional sharpness
        pEnableEmboss=True,             # enable emboss
        pEmbossFactor=.1,              # maximum emboss
        pEnableBrightness=True,         # enable brightness
        pBrightnessFactor=.1,         # maximum +- brightness
        pEnableRandomNoise=True,        # enable random noise
        pMaxRandomNoise=.1,           # maximum random noise strength
        pEnableInvert=False,             # enables color invert
        pEnableContrast=True,           # enable contrast change
        pContrastFactor=.1,            # maximum +- contrast
):
    
    augmentationMap = []
    augmentationMapOutput = []

    if pEnableResizing:
        if random.Random().randint(0, 1)==1:
            randomResizeX = 1 - random.Random().random()*pScaleFactor
        else:
            randomResizeX = 1 + random.Random().random()*pScaleFactor
        if random.Random().randint(0, 1)==1:
            randomResizeY = 1 - random.Random().random()*pScaleFactor
        else:
            randomResizeY = 1 + random.Random().random()*pScaleFactor
        aug = iaa.Resize({"height": randomResizeY, "width": randomResizeX})
        augmentationMap.append(aug)
            
    if pEnableCropping:
        randomCrop2 = random.Random().random()*pCropFactor
        randomCrop4 = random.Random().random()*pCropFactor
        randomCrop1 = random.Random().random()*pCropFactor
        randomCrop3 = random.Random().random()*pCropFactor
        aug = iaa.Crop(percent = (randomCrop1,randomCrop2,randomCrop3,randomCrop4))
        augmentationMap.append(aug)

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

        
    widthFactor = 1
    heightFactor = 1

    arr = numpy.arange(0,len(augmentationMap))
    numpy.random.shuffle(arr)
    
    switchWidthHeight = False
    for i in range(pNbAugmentations):
        augmentationMapOutput.append(augmentationMap[arr[i]])
        if arr[i]==0:
            widthFactor *= randomResizeX
            heightFactor *= randomResizeY
        if arr[i]==1:
            widthFactor *= (1.0 - (randomCrop2 + randomCrop4))
            heightFactor *= (1.0 - (randomCrop1 + randomCrop3))
        if arr[i]==4:
            if randomNumber==1 or randomNumber==3:
                switchWidhtHeight = True
        
    return iaa.Sequential(augmentationMapOutput), widthFactor, heightFactor, switchWidthHeight