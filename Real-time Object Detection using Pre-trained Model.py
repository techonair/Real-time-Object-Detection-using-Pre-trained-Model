# Importing libraries

import numpy as np
import cv2

# Declaring the Pre-trained model

prototxt = 'MobileNetSSD_deploy.prototxt.txt'
model = 'MobileNetSSD_deploy.caffemodel'

# Assigning a confidence threshold for better detection accuracy

confThresh = 0.2

# Declaring the classes

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

# Below code selects random color uniformly and uniquely for different classes

COLORS = np.random.uniform(0,255, size = (len(CLASSES), 3))

# Loading model
print('loading model')
net = cv2.dnn.readNetFromCaffe(prototxt, model)
print("Model Loaded")

# Starting real time video capturing
print('Starting Camera Feed')
cam = cv2.VideoCapture(0)


# initializing loop

while True:

    # reading frames from video and preprocessing
    # -> resizing -> filtering blob
    _,frame = cam.read()
    #frame = cv2.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    imResizeBlob = cv2.resize(frame, (300, 300))
    blob = cv2.dnn.blobFromImage(imResizeBlob, 0.007843, (300, 300), 127.5)
    #cv2.imshow('blob', blob)
    # sending preprocessed frame as input to pretrained neural network
    # forwarding into deep neural net
    # and reshaping it 
    net.setInput(blob)
    detection = net.forward()
    detShape = detection.shape[2]

    # Applying confidence threhold on output detection to classify objects

    for i in np.arange(0, detShape):

        confidence = detection[0,0,i,2]

        if confidence > confThresh:
            idx = int(detection[0,0,i,1])
            print('Class ID: ', detection[0,0,i,1])
            box = detection[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype('int')

            label = "{}: {:.2f}%".format(CLASSES[idx],
				confidence * 100)
                
            cv2.rectangle(frame, (startX, startY), (endX, endY),
				COLORS[idx], 2)
			
            if startY - 15 > 15:
                y = startY - 15
            else:
                startY + 15
            cv2.putText(frame, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()

