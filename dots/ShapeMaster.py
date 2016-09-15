import math
import random
import os
import Image, ImageDraw, ImageFilter, ImageFont
import copy
from euclid import euclid
import datetime

class ShapeMaster:
	def __init__(self, box=[640, 640], shapesize = 0.3, shape= 'circle', sizemeasure='area', density=5, separation=25, colors =[[255, 255, 255]], overlay = False, bgcolor = [0,0,0], outline = [255, 255, 255], control='', drawOutline=False, MIN=.2, MAX=.8, texture=None, sameSize = False):

		#make an output directory
		if not os.path.exists("stimuli"):
			os.mkdir("stimuli")

		self.texture = texture

		self.box = box
		self.logFile = "dot_log_%s.csv" % datetime.datetime.now()
		self.ctl_iters = 1
		self.shape = shape
		#if one item size provided
		#otherwise don't size control
		self.sizemeasure = sizemeasure

		self._setSize(shapesize)

		self.MIN = MIN
		self.MAX = MAX

		self.sameSize = sameSize

		self.overlay = overlay
		self.density = density
		self.separation = separation

		self.colors = colors
		self.bgcolor = bgcolor
		self.outline = outline
		self.control = control
		self.controlValue = False
		self.drawOutline = drawOutline

	def _setSize(self, shapesize):
		if self.sizemeasure == 'area':
			self.shapeSize = self.box[0] * self.box[1] * shapesize
		elif self.sizemeasure == 'perimeter':
			self.shapeSize = self.box[0] + self.box[1] * shapesize


	def _shapeSolver(self, n=1, size=100, control = ''):
		#input - number of dots
		#solver algo
		#1 - calculate average size
		#2 - generate a random value which is a portion of that size
		#3 - randomly determine to add or subtract that
		#4 - perform that operation on the dot
		#5 - repeat until only 1 dot is left
		#6 - make the last dot the necessary size so the size works out

		avg = size / float(n)
		operations = [-1, 1]
		mySizes = []

		if n > 1:
			if self.sameSize:
				mySizes = [avg] * n
			else:
				#make a guess on the average sizes of the n-1 of the items
				for i in range(n-1):
					num = random.uniform(self.MIN, self.MAX)
					operation = random.choice(operations)
					mySizes.append(avg + (operation * num * avg))

				#determine the appropriate size of the nth item
				total = sum(mySizes)
				diff = size - total

				if diff > 0 and diff >= (avg*self.MIN) and diff <= (avg*self.MAX):
					mySizes.append(diff)
				else:
					mySizes = []
					while not mySizes:
						mySizes = self._shapeSolver(n, size)
					mySizes = mySizes[0]

		else:
			mySizes = [size]


		#optional step, control for another size dimension
		controlSizes = []

		#determine the size of the controlled dimension
		for ms in mySizes:
			r = euclid[self.shape]['radius'](ms, self.sizemeasure)
			if control:
				cs = euclid[self.shape][control](r)
			else:
				if self.sizemeasure == 'area':
					cs = euclid[self.shape]['perimeter'](r)
				else:
					cs = euclid[self.shape]['area'](r)
			controlSizes.append(int(cs))

		#if we have a list of items and we want to control a dimension
		if mySizes and control:
			#control value has not been set
			if self.controlValue == False:
				controlSizes = []
				iters = 100
				for i in range(iters):
					mySizes, controlSize = self._shapeSolver(n, size)
					controlSizes.append(controlSize)
				self.controlValue = sum(controlSizes) / iters
				print "CONTROL VALUE : %s" % self.controlValue
				return []
			#controlvalue has been set
			else:
				#% similarity of controlled value
				threshold = 95
				#print self.controlValue, controlSizes
				vals = [self.controlValue, sum(controlSizes)]
				control_ratio =  min(vals) / float(max(vals)) * 100.
				#check and see whether the control dimension is controlled enough (95% threshold)
				#print control_ratio
				#print self.ctl_iters
				#if control_ratio >= threshold or self.ctl_iters <= 1000:
				if control_ratio >= threshold or self.ctl_iters >= 10000:
					#it is, so we're done and we can reset the control value
					self.ctl_iters = 1
					return mySizes, sum(controlSizes)
				else:
					#it isn't
					self.ctl_iters += 1
					return []

		#we have a list of items and don't want to control it
		elif mySizes:
			return mySizes, sum(controlSizes)


	def _generateLists(self, n, control=''):
		#generates the area lists, depending on the ns parameters
		sizeList = self._shapeSolver(n, self.shapeSize, control=control)[0]

		self.controlValue = False
		return sizeList

	def shapeArranger(self, n):
		#1 - place a dot box in a random location which does not overlap the edges
		#2 - place a dot in a random location which does not overlap the edges or any other dot boxes
		#3 - repeat until no dots are left

		goodList = 0
		breaks = 0

		sizeList = self._generateLists(n, self.control)

		if len(sizeList) > 1:

			while not goodList:
				shapeBoxes = []
				count = 1
				shapesizes = copy.deepcopy(sizeList)

				while len(shapesizes):
					a = shapesizes[-1]
					r = int(euclid[self.shape]['radius'](a, self.sizemeasure))
					d = int(r * 2)
					quit = 0
					reps = 1

					#if we've broken the cycle more than 10 times, we should regenerate the area list, 'cause this obviously ain't workin'
					if breaks > 9:
						sizeList, sepShapes = self._generateLists(n, self.control)
						breaks = 0
						shapeBoxes = []
						shapesizes = copy.deepcopy(sizeList)

					while not quit:
						reps = reps + 1
						#if we've tried this too many times, restart the process...
						if reps > 100000:
							#delete the list of boxes
							#put the area back into the list of areas
							shapesizes = copy.deepcopy(sizeList)
							#and break the loop
							breaks = breaks + 1
							shapeBoxes = []
							quit = 1

						x = int(random.uniform(r + self.density, self.box[0] - r - self.density))
						y = int(random.uniform(r + self.density, self.box[1] - r - self.density))

						shapeBox = [x, y, r, a]

						#if there are no dots on the screen, place the current dot on the screen and proceed to the next placement
						if count == 1:
							shapeBoxes.append(shapeBox)
							shapesizes.pop()
							quit = 1
						#otherwise check against the existing list of dots
						else:
							bad = 0
							for box in shapeBoxes:
								minRadius = r + box[2] + 5 + self.separation
								x2 = box[0]
								y2 = box[1]

								ax = abs(x - x2)
								by = abs(y - y2)

								cSquare = ax**2 + by**2
								c = (cSquare ** 0.5)

								if c < minRadius:
									bad = 1

							if not bad:
								shapeBoxes.append(shapeBox)
								goodList = 1
								shapesizes.pop()
								quit = 1
					count = count + 1
		else:
			a = sizeList[0]
			r = int(euclid['circle']['radius'](a, self.sizemeasure))

			x = int(random.uniform(r + self.density, self.box[0] - r - self.density))
			y = int(random.uniform(r + self.density, self.box[1] - r - self.density))

			shapeBoxes = [[x, y, r, a]]

		self.shapeBoxes = shapeBoxes

	def drawSingle(self, name=""):
		image = Image.new("RGB", self.box, self.bgcolor)
		draw = ImageDraw.Draw(image)

		if name:
			fname = "%s" % name

		else:
			fname = "%s_%s_S%s" % (name, count)

		f = open('stimuli/%s.svg' % fname, 'w')
		f.write('<svg xmlns="http://www.w3.org/2000/svg" width="%s" height="%s">\n' % (self.box[0]/2, self.box[1]/2))

		coords = []

		for d in self.shapeBoxes:
			f.write('\t<circle cx="%s" cy="%s" r="%s" fill="blue" />\n' % (d[0]/2., d[1]/2., d[2]/2.))
			coords.append([d[0], d[1], d[2]])
			box1 = [d[0] - d[2], d[1] - d[2], d[0] + d[2], d[1] + d[2]]
			draw.ellipse(box1, fill = self.colors[0])

		image.save("stimuli/%s.png" % fname, "PNG")
		f.write('</svg>')
		f.close()

		return coords


	def _printLog(self, fname="shapes"):
		if os.path.exists(self.logFile):
			f = open(self.logFile, "a")
		else:
			f = open(self.logFile, "w")
			f.write("file,shape,n1,n2,ratio,1/ratio,area_n1,area_n2,area_ratio,per_n1,per_n2,per_ratio\n")

		log = "%s, %s, %s, %s" % (fname, self.shape, self.ratio_log, self.size_log)

		f.write(log + "\n")
		f.close()
