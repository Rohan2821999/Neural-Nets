#dot_demo.py
import numpy
from ShapeMaster import ShapeMaster

bgcolor = (0, 0, 0)
colors = [(255, 255, 255), (255,0,0)]

ns = range(1, 10)
box = (50, 50)
shape = 'circle'
area = 0.1

s = ShapeMaster(box, area, shape=shape, sizemeasure = 'area', colors = colors, bgcolor = bgcolor, outline=(0, 0, 0), drawOutline=True, separation = 5, density = 1)
s._setSize(area)

for n in ns:
	for instance in range(3):
		s.shapeArranger(n)
		s.drawSingle("%s_%s" % (n, instance + 1))
