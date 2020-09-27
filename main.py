import os
import cv2
#disable tensor flow output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
    tf.config.experimenta.set_memory_growth((tf.config.experimental.list_physical_devices('GPU'), True))

from absl import app
import util as util
from tensorflow.python.saved_model import tag_constants
import cv2
import random
import colorsys
from PIL import Image

import numpy as np
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def main(_argv):
    #Create video source
    cap = cv2.VideoCapture(0)
    frameNumber = 0

    #Load saved model
    savedModelLoded = tf.saved_model.load('./custom-416', tags =[tag_constants.SERVING])
    infer = savedModelLoded.signatures['serving_default']

    

    while True:
        #Start Video Capture
        ret, frame = cap.read()
        #Convert color space
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frameNumber +=1
        image = Image.fromarray(frame)

        #Create frame specifics
        frameSize = frame.shape[:2]
        #Resize to 416
        imageData = cv2.resize(frame, (416,416))
        imageData = imageData/255
        imageData = imageData[np.newaxis, ...].astype(np.float32)

        batchData = tf.constant(imageData)
        predictedBBox = infer(batchData)
        for key, value in predictedBBox.items():
            boxes = value[:,:,0:4]
            predictedConf=value[:,:,4:]
        
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes,(tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(predictedConf, (tf.shape(predictedConf)[0], -1, tf.shape(predictedConf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.5
        )
    
         #Create bounding boxed from normalized coordinats
        o_Height, o_Width, __ignore = frame.shape
        bboxes = util.formatBoxes(boxes.numpy()[0], o_Height, o_Width)

        predictedBBox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

        className = 'licensePlate'

        #Crop each detection and save it as an image
        cropRate = 125 #Every 125 frames
        cropPath = os.path.join(os.getcwd(), 'detections', 'croped')
        #Create new directory first time
        try: 
            os.mkdir(cropPath)
        except FileExistsError:
            pass 

        if frameNumber % cropRate == 0:
            try:
                os.mkdir(
                    os.path.join(cropPath, 'frame_' + str(frameNumber))
                )
            except FileExistsError:
                pass

            #crop each detection, then save as new image
            boxes, scores, classes, numObjects = predictedBBox
            tmpIMG = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            #create dictonary to hold amount of objects for image name
            countObj = 0

            for i in range(numObjects):
                countObj+=1
                #get box coordinates
                xMin, yMin, xMax, yMax = boxes[i]
                #crop detection, remove additional pixels around edges
                croppedImg = tmpIMG[int(yMin)-5:int(yMax)+5, int(xMin)-5:int(xMax)+5]

                #Construct image, join it to path to save crop
                cv2.imwrite(os.path.join(os.path.join(cropPath, 'frame_' + str(frameNumber)) + className + '_' + str(countObj) + '.png', croppedImg))
        else:
            pass

        #Draw Bounding Box

        imageH, imageW, __ignore = frame.shape


        outBoxes, outScores, outClasses, numBoxes= predictedBBox

        for i in range(numBoxes):
            if int(outClasses[i]) < 0 or int(outClasses[i]) > 1: continue
            coo = outBoxes[i]
            fontScale = 0.5
            score = outScores[i]
            Hratio = int(imageH/25)
            #find plate number
            plateNumber = util.getPlateNumber(frame, coo)
            if plateNumber != None:
                cv2.putText(frame, plateNumber, (int(coo[0]), int(coo[1]-Hratio)), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255,255,0), 2)
            
            bboxThicc = int(0.6 * (imageH+imageW)/600)
            cOne, cTwo = ((coo[0], coo[1]), (coo[2], coo[3]))
            cv2.rectangle(frame, cOne, cTwo, (255,0,0), bboxThicc)

            #Print info
            print("Object: {}, Confidence: {:.2f}, BBox Coords (xMin, yMin, xMax, yMax): {}, {}, {}, {}".format(className, score, coo[0], coo[1], coo[2], coo[3]))

        result = np.asarray(frame)

        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass




