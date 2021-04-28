import numpy
import skimage

def RCNNConvertInputImage(pImageData):
    
    if pImageData.ndim == 2:
        pImageData = skimage.color.gray2rgb(pImageData)
    if pImageData.ndim == 3:
        if pImageData.shape[0] < pImageData.shape[2]:
            new_pImageData = numpy.zeros((pImageData.shape[1], pImageData.shape[2], 3), numpy.uint8)
            for k in range(pImageData.shape[0]):
                new_pImageData[:,:,k] = (pImageData[k, :, :]).astype('uint8')
            for k in range(pImageData.shape[0], 3):
                new_pImageData[:,:,k] = (pImageData[pImageData.shape[0]-1, :, :]).astype('uint8')
            pImageData = new_pImageData
        elif pImageData.shape[2] < 3:
            new_pImageData = numpy.zeros((pImageData.shape[0], pImageData.shape[1], 3), numpy.uint8)
            for k in range(pImageData.shape[2]):
                new_pImageData[:,:,k] = (pImageData[:, :, k]).astype('uint8')
            for k in range(pImageData.shape[2], 3):
                new_pImageData[:,:,k] = (pImageData[:, :, pImageData.shape[2]-1]).astype('uint8')
            pImageData = new_pImageData
            
    return pImageData

def MergeMasks(pMasks):
    if pMasks.ndim < 3:
        raise ValueError("Invalid masks")

    maskCount = pMasks.shape[2]
    width = pMasks.shape[1]
    height = pMasks.shape[0]
    mask = numpy.zeros((height, width), numpy.uint16)

    for i in range(maskCount):
        mask[:,:] = numpy.where(pMasks[:,:,i] != 0, i+1, mask[:,:])

    return mask


def PadImageR(pImageData, pRatio):
    width = pImageData.shape[1]
    height = pImageData.shape[0]

    x = int(float(width) * float(pRatio))
    y = int(float(height) * float(pRatio))

    image = PadImageXY(pImageData, x, y)
    return image, (x, y)


def PadImageXY(pImageData, pX, pY):
    width = pImageData.shape[1]
    height = pImageData.shape[0]

    paddedWidth = width + 2 * pX
    paddedHeight = height + 2 * pY


    if pImageData.ndim > 2:
        count = pImageData.shape[2]
        image = numpy.zeros((paddedHeight, paddedWidth, count), pImageData.dtype)
        for c in range(count):
            image[:, :, c] = numpy.lib.pad(pImageData[:, :, c], ((pY, pY), (pX, pX)), "reflect")

    else:
        image = numpy.lib.pad(pImageData, ((pY, pY), (pX, pX)), "reflect")

    return image
