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
from tkinter import *
from PIL import Image, ImageTk
from tkinter import filedialog
import imageio


class Window(Frame):
	def __init__(self, master=None):
		Frame.__init__(self, master)

		self.master = master
		self.pos = []
		self.master.title("GUI")
		self.pack(fill=BOTH, expand=1)

		self.line_point = -1
		self.frame_height = 400
		self.saveno = 1

		menu = Menu(self.master)
		self.master.config(menu=menu)

		file = Menu(menu)
		file.add_command(label="Open", command=self.open_file)
		file.add_command(label="Exit", command=self.client_exit)
		menu.add_cascade(label="File", menu=file)
		
		analyze = Menu(menu)
		analyze.add_command(label="Region of Interest", command=self.regionOfInterest)
		menu.add_cascade(label="Analyze", menu=analyze)

		self.filename = "sample_data/home.jpg"
		self.imgSize = Image.open(self.filename)
		self.imgSize = self.imgSize.resize( (int(self.imgSize.size[0]*self.frame_height/self.imgSize.size[1]), self.frame_height), Image.ANTIALIAS)
		self.tkimage =  ImageTk.PhotoImage(self.imgSize)
		self.w, self.h = (1366, 768)
		
		self.canvas = Canvas(master = root, width = int(self.imgSize.size[0]*self.frame_height/self.imgSize.size[1]), height=self.frame_height)
		self.canvas.create_image(20, 20, image=self.tkimage)
		self.canvas.pack()

	def open_file(self):
		self.filename = filedialog.askopenfilename()

		cap = cv2.VideoCapture(self.filename)

		reader = imageio.get_reader(self.filename)
		fps = reader.get_meta_data()['fps'] 

		ret, image = cap.read()
		cv2.imwrite('output/temp/preview.jpg', image)

		self.show_image('output/temp/preview.jpg')


	def show_image(self, frame):

		self.imgSize = Image.open(frame)
		self.imgSize = self.imgSize.resize( (int(self.imgSize.size[0]*self.frame_height/self.imgSize.size[1]), self.frame_height), Image.ANTIALIAS)
		self.tkimage =  ImageTk.PhotoImage(self.imgSize)
		self.w, self.h = (1366, 768)

		self.canvas.destroy()
		self.canvas = Canvas(master = root, height = self.frame_height, width=int(self.imgSize.size[0]*self.frame_height/self.imgSize.size[1]))
		self.canvas.create_image(0, 0, image=self.tkimage, anchor='nw')
		self.canvas.pack()

	def regionOfInterest(self):
		root.config(cursor="plus") 
		self.canvas.bind("<Button-1>", self.imgClick) 

	def client_exit(self):
		exit()

	def imgClick(self, event):

		if self.line_point < 0:
			x = int(self.canvas.canvasx(event.x))
			y = int(self.canvas.canvasy(event.y))
			self.line_point=y

		if self.line_point >= 0:
			#unbinding action with mouse-click
			self.canvas.unbind("<Button-1>")
			img = cv2.imread('output/temp/preview.jpg')
			cv2.imwrite('output/temp/copy.jpg', img)
			self.show_image('output/temp/copy.jpg')
			
			#image processing
			self.main_process()

			for i in self.pos:
				self.canvas.delete(i)

	def main_process(self):
		Base={
			"max_disappear": 30,

			"max_distance": 200,

			"track_object": 4,

			"confidence": 0.4,

			"frame_height": 400,

			"line_point" : 125,

			"display": "true",

			"model_path": "MobileNetSSD_deploy.caffemodel",

			"prototxt_path": "MobileNetSSD_deploy.prototxt",

			"output_path": "output",

			"csv_name": "log.csv"
		}

		CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
			"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
			"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
			"sofa", "train", "tvmonitor"]

		print("[INFO] loading model...")
		net = cv2.dnn.readNetFromCaffe(Base["prototxt_path"],
			Base["model_path"])


		print("[INFO] warming up camera...")
		vs = cv2.VideoCapture(self.filename)

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
			minut=ts.minute

			if frame is None:
				break

			frame = imutils.resize(frame, height=Base["frame_height"])
			rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

			if W is None or H is None:
				(H, W) = frame.shape[:2]

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
							if CLASSES[idx] != "bus":
								if CLASSES[idx] != "motorbike":
									continue

						box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
						(startX, startY, endX, endY) = box.astype("int")

						tracker = dlib.correlation_tracker()
						rect = dlib.rectangle(int(startX), int(startY), int(endX), int(endY))
						tracker.start_track(rgb, rect)
						cv2.rectangle(frame, (startX, startY), (endX, endY), (0,225,0), 4)
						trackers.append(tracker)

			else:
				for tracker in trackers:
					tracker.update(rgb)
					pos = tracker.get_position()

					startX = int(pos.left())
					startY = int(pos.top())
					endX = int(pos.right())
					endY = int(pos.bottom())
					cv2.rectangle(frame, (startX, startY), (endX, endY), (0,225,0), 4)
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
						if minut%2==0:
							if not to.belowline:
								if(centroid[1] < self.line_point):
									to.belowline = "F"
								else:
									to.belowline = "T"

							else:
								if(to.belowline == "F" and centroid[1] > self.line_point):
									if not to.savethefile:
										#crop = frame[startX:endX, startY:endY]
										cv2.imwrite('output/violation'+str(self.saveno)+'.jpg', frame)
										to.savethefile = 1
										self.saveno += 1
									cv2.circle(frame, (centroid[0]+10, centroid[1]), 4,
									(0, 0, 255), -1)

						else:
							if to.belowline:
								to.belowline = None
							

					elif(to.direction<0):
						tet = "up"
						cv2.putText(frame, tet, (centroid[0] - 10, centroid[1] - 20)
							, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			
					elif(to.direction==0):
						tet = "stationary"
						cv2.putText(frame, tet, (centroid[0] - 10, centroid[1] - 20)
							, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

				trackableObjects[objectID] = to

				text = "ID {}".format(objectID)
				cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10)
					, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
				cv2.circle(frame, (centroid[0], centroid[1]), 4,
					(0, 255, 0), -1)
				if minut%2==0:
					cv2.line(frame, (0, self.line_point), (2000, self.line_point), (0,0,255), 4)
				else:
					cv2.line(frame, (0, self.line_point), (2000, self.line_point), (0,255,0), 4)

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

root = Tk()
app = Window(root)
root.geometry("%dx%d"%(535, 380))
root.title("Traffic Violation")

root.mainloop()