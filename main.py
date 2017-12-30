import numpy as np
import cv2

# 讀入照片
def readPhoto(name):
    return cv2.imread(name)


# 儲存照片
def savePhoto(name, im):
    cv2.imwrite(name + '.png', im)


# 取得並儲存灰階照片
def readGrayImage(name):
    grayImage = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    savePhoto('gray_image', grayImage)
    return grayImage


# Sobel Filter
def sobelFilter(image):
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))
    sobelCombined = cv2.bitwise_or(sobelX, sobelY)
    savePhoto('sobel_filter_image', sobelCombined)


def laplacianMask(image):
    lap = cv2.Laplacian(image, cv2.CV_64F)
    lap = np.uint8(np.absolute(lap))
    savePhoto('laplacian_mask_image', lap)
    return lap


def lap_enhance(source, lapacian):
    height = source.shape[0]
    width = source.shape[1]
    # height, width, _ = source.shape
    newImage = np.zeros((height, width, 3), np.uint8)
    for x in range(width):
        for y in range(height):
            value = source[y][x] + lapacian[y][x]
            if value > 255:
                value = 255
            else:
                value = int(value)
            newImage[y][x] = value#source[y][x] + lapacian[y][x] #int(value)
    savePhoto('lap_enhance_image', newImage)
    return newImage


def convolution():
    image = cv2.imread('sobel_filter_image.png')
    blurred = cv2.blur(image, (3, 3))
    savePhoto('convolution_image', blurred)
    return blurred


def normalization(source, blur):
    height = source.shape[0]
    width = source.shape[1]
    # height, width, _ = source.shape
    newImage = np.zeros((height, width, 3), np.uint8)
    for x in range(width):
        for y in range(height):
            newImage[y][x] = blur[y][x] / 255 * source[y][x]
    savePhoto('normalization_image', newImage)
    return newImage


def enhancementImage(source, laplacianMask, normalization):
    height = source.shape[0]
    width = source.shape[1]
    # height, width, _ = source.shape
    newImage = np.zeros((height, width, 3), np.uint8)
    for x in range(width):
        for y in range(height):
            # value = normalization[y][x] * laplacianMask[y][x] + source[y][x]
            value = normalization[y][x] + source[y][x]
            # print(value)
            newImage[y][x] = value
    savePhoto('FinalEnhancementImage', newImage)

# 0
# sourceImage = readPhoto('source.jpeg')
sourceImage = readGrayImage('lena_color.png')
# 1
laplacianMaskImage = laplacianMask(sourceImage)
# 2
laplacianEnhancementImage = lap_enhance(sourceImage, laplacianMaskImage)
# 3
sobelFilterImage = sobelFilter(sourceImage)
# 4
blurImage = convolution()
# 5
normalizationImage = normalization(blurImage, laplacianEnhancementImage)
# Final
enhancementImage(sourceImage, laplacianMaskImage, normalizationImage)
