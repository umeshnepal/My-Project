# import the necessary packages
import numpy as np

class TrackableObject:
	def __init__(self, objectID, centroid):
		# store the object ID, then initialize a list of centroids
		# using the current centroid
		self.objectID = objectID
		self.centroids = [centroid]
		self.lastPoint = False
	

		# initialize two booleans, (1) used to indicate if the
		# object's speed has already been estimated or not, and (2)
		# used to indidicate if the object's speed has been logged or
		# not
		self.estimated = False
		self.logged = False

		# initialize the direction of the object
		self.direction = None
		self.belowline = None
		self.savethefile = None

	def calculate_speed(self, estimatedSpeeds):
		# calculate the speed in KMPH and MPH
		self.speedKMPH = np.average(estimatedSpeeds)
		MILES_PER_ONE_KILOMETER = 0.621371
		self.speedMPH = self.speedKMPH * MILES_PER_ONE_KILOMETER