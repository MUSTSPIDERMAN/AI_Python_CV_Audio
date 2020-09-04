
#python3.7 detect4.py --model liveness.model --le le.pickle --detector face_detector --shape-predictor shape_predictor_68_face_landmarks.dat --yolo yolo-coco --tracker csrt

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

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to trained model")
ap.add_argument("-l", "--le", type=str, required=True,
	help="path to label encoder")
ap.add_argument("-d", "--detector", type=str, required=True,
	help="path to OpenCV's deep learning face detector")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
ap.add_argument("-y", "--yolo", required=True,
    help="base path to YOLO directory")
#ap.add_argument("-c", "--confidence", type=float, default=0.5,
  #  help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
    help="threshold when applyong non-maxima suppression")
ap.add_argument("-w", "--tracker", type=str, default="kcf",
help="OpenCV object tracker type")
args = vars(ap.parse_args())

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# load our serialized face detector from disk
print("[INFO] loading face detector...")
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])
modelPath = os.path.sep.join([args["detector"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load the liveness detector model and label encoder from disk
print("[INFO] loading liveness detector...")
model = load_model(args["model"])
le = pickle.loads(open(args["le"], "rb").read())

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


# initialize the video stream and allow the camera sensor to warmup
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
print("[INFO] loading YOLO from disk...")
net2 = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

class_name = "/Users/xiuliming/Documents/MachineLearning/face_recognition/face_recognition-master/invaders"

index = 1

video_capture = cv2.VideoCapture(0)

Spiderman_image = face_recognition.load_image_file("Spiderman.jpeg")
Spiderman_face_encoding = face_recognition.face_encodings(Spiderman_image)[0]

UncleSam_image = face_recognition.load_image_file("UncleSam.jpg")
UncleSam_face_encoding = face_recognition.face_encodings(UncleSam_image)[0]

# JamesZaworski_image = face_recognition.load_image_file("JamesZaworski.jpg")
# JamesZaworski_face_encoding = face_recognition.face_encodings(JamesZaworski_image)[0]

Michael_image = face_recognition.load_image_file("Michael.jpg")
Michael_face_encoding = face_recognition.face_encodings(Michael_image)[0]

    #longlong_image = face_recognition.load_image_file("longlong.jpg")
    #long_face_encoding = face_recognition.face_encodings(longlong_image)[0]

#MyLOVE_image = face_recognition.load_image_file("MyLOVE.jpg")
#MyLOVE_face_encoding = face_recognition.face_encodings(MyLOVE_image)[0]

#know_names = []
#know_encodings = []

#for filename in os.listdir(r"./" + "invaders"):
#    img = cv2.imread("invaders" + "/" + filename)
#    encoding = face_recognition.face_encodings(img)[0]
#    know_encodings.append(encoding)
#    know_names.append(filename)

known_face_encodings = [
    # obama_face_encoding,
    # biden_face_encoding,
    Spiderman_face_encoding,
    # MARBEL_face_encoding,
    #  SUNXIUJUAN_face_encoding,
    UncleSam_face_encoding
    # JamesZaworski_face_encoding,
                        # Michael_face_encoding,
                        # long_face_encoding
                        #  MyLOVE_face_encoding
    #  XIULIMING_face_encoding
    # Jim_face_encoding
]
known_face_names = [
    # "Barack Obama",
    # "Joe Biden",
    "Spiderman",
    #  "MARBEL"
    # "SUNXIUJUAN",
      "UncleSam"
    # "JamesZaworski",
                    # "Michael"
                    #  "longlong"
                    #  "MyLOVE"
    #  "XIULIMING"
    # "Jim"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    # loop over the frames from the video stream
while True:
    frame = vs.read()

    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    #rgb_small_frame = small_frame[:, :, ::-1]


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
            face = cv2.resize(face, (64, 64))
            face = face.astype("float") / 255.0
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)

            # pass the face ROI through the trained liveness detector
            # model to determine if the face is "real" or "fake"
            preds = model.predict(face)[0]
            j = np.argmax(preds)
            label = le.classes_[j]
            if label == "real":
                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                # if label == "real: 1.0000":
                #      break
                cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 255, 0), 2)
            else:
                # draw the label and bounding box on the frame
                label = "{}: {:.4f}".format(label, preds[j])
                # if label == "real: 1.0000":
                #      break
                cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.rectangle(frame, (startX, startY), (endX, endY),(0, 0, 255), 2)
                

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


    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    # do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()


#################################################################################################
OPENCV_OBJECT_TRACKERS = {
       "csrt": cv2.TrackerCSRT_create,
       "kcf": cv2.TrackerKCF_create,
       "boosting": cv2.TrackerBoosting_create,
       "mil": cv2.TrackerMIL_create,
       "tld": cv2.TrackerTLD_create,
       "medianflow": cv2.TrackerMedianFlow_create,
       "mosse": cv2.TrackerMOSSE_create
   }
trackers = cv2.MultiTracker_create()



while True:

        # Grab a single frame of video
        #ret, frame = video_capture.read()
    
    ret, frame = video_capture.read()
   #@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
   
    (success, boxes) = trackers.update(frame)
    for box in boxes:
        (x, y, w, h) = [int(v) for v in box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the output frame
    key = cv2.waitKey(1) & 0xFF

    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
        trackers.add(tracker, frame, box)#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # land_mark
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

    (H, W) = frame.shape[:2]
    
    # determine only the *output* layer names that we need from YOLO
    ln = net2.getLayerNames()
    ln = [ln[i[0] - 1] for i in net2.getUnconnectedOutLayers()]
    
    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net2.setInput(blob)
    start = time.time()
    layerOutputs = net2.forward(ln)
    end = time.time()
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
        
    boxes = []
    confidences = []
    classIDs = []
    
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
                                         
                                       
            if confidence > args["confidence"]:
                                      
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                                             
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                                             
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"],
                        args["threshold"])

   



    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    rgb_small_frame = small_frame[:, :, ::-1]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # land_mark
# detect faces in the grayscale frame
    rects = detector(gray, 0)


#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    if process_this_frame:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            # if name == "Spiderman":
            #  print ('\7')
            # print ('\a')
            # os.system('play --no-show-progress --null --channels 1 synth %s sine %f' %(duration, freq))

            # print('Spiderman: %f ' %(Spiderman_face_encoding))
            # print('Spiderman  position:%d %d %d %d' %( top, right, bottom, left))
           # if name == "Unknown":
            #    matches = face_recognition.compare_faces(know_encodings, face_encoding)
             #   face_distances = face_qrecognition.face_distance(know_encodings, face_encoding)
              #  best_match_index = np.argmin(face_distances)
               # if matches[best_match_index]:
                #    name = know_names[best_match_index]
            if name == "Unknown":
                known_face_encodings.append(face_encoding)
                known_face_names.append(index)
                index +=1
           # if matches[best_match_index]:
            #    name = know_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame
    index2 = 1
    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if name == "Spiderman":
            print(' %s position:%d %d %d %d' % (name, top, right, bottom, left))
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 0, 255), 1)


        elif name == "Unknown":
            print('non-admin position:%d %d %d %d' % (top, right, bottom, left))
            print('\7')
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            if index2 < 200:
                cv2.imwrite("%s/%d.jpeg" % (class_name, index2),
                            cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA))
            index2 += 1
        else :
            print('%s position:%d %d %d %d' % ( name ,top, right, bottom, left))
            print('\7')
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, str(  name), (left + 6, bottom - 6), font, 2.0, (255, 0, 255), 0)


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

    if len(idxs) > 0:
                        # loop over the indexes we are keeping
       for i in idxs.flatten():
                            # extract the bounding box coordinates
           (x, y) = (boxes[i][0], boxes[i][1])
           (w, h) = (boxes[i][2], boxes[i][3])
           color = [int(c) for c in COLORS[classIDs[i]]]
           cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
           text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
           cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, color, 2)



    cv2.imshow('Video', frame)
    cv2.waitKey(1)


    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



video_capture.release()
cv2.destroyAllWindows()
