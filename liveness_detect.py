from imutils.video import VideoStream
from imutils.video import FPS
from keras.preprocessing.image import img_to_array
from keras.models import load_model
#import numpy as np
import argparse
import imutils
import pickle
import time
#import cv2

import face_recognition
import cv2
import numpy as np
import time
import pickle
import os

from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

"""
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())
"""
args={'confidence':0.5,'threshold':0.3}
def liveness_demo():
    #protoPath = os.path.sep.join(["face_detector ", "deploy.prototxt"])
    protoPath = "face_detector/deploy.prototxt"
    #modelPath = os.path.sep.join(["face_detector ", "res10_300x300_ssd_iter_140000.caffemodel"])
    modelPath = "face_detector/res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # load the liveness detector model and label encoder from disk
    print("[INFO] loading liveness detector...")
    model = load_model("liveness.model")
    le = pickle.loads(open("le.pickle", "rb").read())

    print("[INFO] loading facial landmark predictor...")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

    while True:
        frame = vs.read()
        # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # rgb_small_frame = small_frame[:, :, ::-1]

        #################################################################################################

        #  frame = vs.read()
        frame = imutils.resize(frame, width=600)
        #################################################################################################
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # land_mark
        # detect faces in the grayscale frame
        rects = detector(gray, 0)
        #################################################################################################
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the
            # prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections
            if confidence > args["confidence"]:
                # compute the (x, y)-coordinates of the bounding box for
                # the face and extract the face ROI
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the detected bounding box does fall outside the
                # dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w, endX)
                endY = min(h, endY)

                # extract the face ROI and then preproces it in the exact
                # same manner as our training data
                face = frame[startY:endY, startX:endX]
                try:
                    face = cv2.resize(face, (64, 64))
                except:
                    pass
                face = face.astype("float") / 255.0
                face = img_to_array(face)
                face = np.expand_dims(face, axis=0)

                # pass the face ROI through the trained liveness detector
                # model to determine if the face is "real" or "fake"
                preds = model.predict(face)[0]
                j = np.argmax(preds)
                label = le.classes_[j]
                print(label)
                if label == "real":
                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f}".format(label, preds[j])
                    # if label == "real: 1.0000":
                    #      break
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                else:
                    # draw the label and bounding box on the frame
                    label = "{}: {:.4f}".format(label, preds[j])
                    # if label == "real: 1.0000":
                    #      break
                    cv2.putText(frame, label, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)

        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

        #cv2.namedWindow("liveness",cv2.WINDOW_NORMAL)
        cv2.imshow("liveness", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            cv2.destroyAllWindows()
            vs.stop()
            break

        # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()



