from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.io import TempFile
from imutils.video import FPS
from datetime import datetime
from threading import Thread
import numpy as np
import argparse
import dropbox
import imutils
import dlib
import time
import cv2
import os


Base={
    "max_disappear": 10,

    "max_distance": 175,

    "track_object": 4,

    "confidence": 0.4,

    "frame_height": 400,

    "speed_estimation_zone": {"A": 120, "B": 160, "C": 200, "D": 240},

    "line_point" : 125,

    "distance": 16,

    "speed_limit": 17,

    "display": "true",

    "model_path": "MobileNetSSD_deploy.caffemodel",

    "prototxt_path": "MobileNetSSD_deploy.prototxt",

    "use_dropbox": "false",

    "dropbox_access_token": "YOUR_DROPBOX_APP_ACCESS_TOKEN",

    "output_path": "output",

    "csv_name": "log.csv",

    "input_path":"sample_data/video4.mp4"
}

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(Base["prototxt_path"],
    Base["model_path"])


print("[INFO] warming up camera...")
vs = cv2.VideoCapture(Base["input_path"])

H = None
W = None

ct = CentroidTracker(maxDisappeared=Base["max_disappear"],
    maxDistance=Base["max_distance"])
trackers = []
trackableObjects = {}

totalFrames = 0

logFile = None

points = [("A", "B"), ("B", "C"), ("C", "D")]

fps = FPS().start()

while True:
    ret, frame  = vs.read()
    ts = datetime.now()
    newDate = ts.strftime("%m-%d-%y")

    if frame is None:
        break

    frame = imutils.resize(frame, height=Base["frame_height"])
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
        meterPerPixel = Base["distance"] / W

    rects = []

    if totalFrames % Base["track_object"] == 0:
        trackers = []

        blob = cv2.dnn.blobFromImage(frame, size=(300, 300),
            ddepth=cv2.CV_8U)
        net.setInput(blob, scalefactor=1.0/127.5, mean=[127.5,
            127.5, 127.5])
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > Base["confidence"]:
                idx = int(detections[0, 0, i, 1])

                if CLASSES[idx] != "car":
                    continue

                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")

                tracker = dlib.correlation_tracker()
                rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
                tracker.start_track(rgb, rect)

                trackers.append(tracker)

    else:
        for tracker in trackers:
            tracker.update(rgb)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    objects = ct.update(rects)

    for (objectID, centroid) in objects.items():
        to = trackableObjects.get(objectID, None)

        if to is None:
            to = TrackableObject(objectID, centroid)

        elif not to.estimated:
            
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.direction = direction
            if(to.direction>0):
                tet = "down"
                cv2.putText(frame, tet, (centroid[0] - 10, centroid[1] - 20)
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if not to.belowline:
                    if(centroid[1] < Base["line_point"]):
                        to.belowline = "F"
                    else:
                        to.belowline = "T"

                else:
                    if(to.belowline == "F" and centroid[1] > Base["line_point"]):
                        cv2.circle(frame, (centroid[0]+10, centroid[1]), 4,
                        (0, 0, 255), -1)

            elif(to.direction<0):
                tet = "up"
                cv2.putText(frame, tet, (centroid[0] - 10, centroid[1] - 20)
                    , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    


        trackableObjects[objectID] = to

        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
            , cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4,
            (0, 255, 0), -1)
        cv2.line(frame, (0, Base["line_point"]), (2000, Base["line_point"]), (0,255,0), 4)


    if Base["display"]=="true":
        cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    
    totalFrames += 1
    fps.update()

fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
vs.release()