import cv2
import random
import colorsys
import numpy as np
import tensorflow as tf
import pytesseract
import re
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


#recognize license plate numbers using Tesseract
def getPlateNumber(image, coordinates):
    xMin, yMin, xMax, yMax = coordinates
    #BBOX
    box = image[int(yMin)-5:int(yMax)+5, int(xMin)-5:int(xMax)+5]

    #greyscale
    grey = cv2.cvtColor(box, cv2.COLOR_RGB2GRAY)

    #resize 3X
    grey = cv2.resize(grey, None, fx=3, fy=3, interpolation= cv2.INTER_CUBIC)

    #Blur image
    blur = cv2.GaussianBlur(grey, (5,5), 0)

    #Thresholding using Otsus
    ret, threshold = cv2.threshold(grey, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))

    #dialation
    dialated = cv2.dilate(threshold, kernel, iterations = 1)

    #Generate contours
    try:
        contours, hierarchy = cv2.findContours(dialated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    except:
        retImg, contours, hierarchy = cv2.findContours(dialated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #Read left -> right

    sortedContours = sorted(contours, key=lambda CTR: cv2.boundingRect(CTR)[0])

    #create copy of greyscale
    greyCopy = grey.copy()

    plateNumber = ""

    #Loop through contours, matching each to number or letter
    for cnt in sortedContours:
        x, y, w, h = cv2.boundingRect(cnt)
        Height, Width = greyCopy.shape

        #Rules to get rid of bad contours

        
        if (Height / float(h) >6): #Height not tall enought
            continue
        if (h/float(w) < 1.5): #Aspect ratio improper
            continue
        if (Width/float(w) > 15): #Not wide enough
            continue
        if ((h*w) < 100): #Too small 
            continue

        #draw Rect
        rect = cv2.rectangle(greyCopy, (x, y), (x+w, y+h), (160, 32, 240), 2)

        #Get Char region
        region = threshold[y-5:y+h+5, x-5:x+w+5]

        #Invert colors
        region = cv2.bitwise_not(region)

        #Blur Char region
        if not region.empty():
            region = cv2.medianBlur(region, 5)

        #Extract text
        try:
            text = pytesseract.image_to_string(region,
                config='-c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ --psm 8 --oem 3')
            #Remove whitespace
            plateNumber+= re.sub('[\W_]+', '', text)
        except:
            text = None
    if plateNumber != None:
        print("License Plate : ", plateNumber)

    cv2.imshow("Character's Segmented", greyCopy)
    return plateNumber




def formatBoxes(bboxes, Height, Width):
    for box in bboxes:
        yMin = int(box[0]*Height)
        xMin = int(box[1]*Width)
        yMax = int(box[2]*Height)
        xMax = int(box[3]*Width)
        box[0], box[1], box[2], box[3] = xMin, yMin, xMax, yMax
    return bboxes 