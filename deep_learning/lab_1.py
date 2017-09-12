from PIL import Image, ImageDraw
import numpy as np
from matplotlib.pyplot import imshow
from IPython.display import display

class generate_rand_image():
	def __init__(self,width,height):
		self.width = width
		self.height = height
		mode = 'RGBA'
		#rgba values------------------
		r = np.random.randint(255)
		g = np.random.randint(255)
		b = np.random.randint(255)
		a = np.random.randint(255)
		#----------------------------

		self.pic = Image.new(mode,(self.width,self.height),color=(r,g,b,a))
		self.drawing = ImageDraw.Draw(self.pic,mode='RGBA')
	
	def draw(self,shape):
		#pixel coordinates--------
		x1 = np.random.randint(self.width)
		y1 = np.random.randint(self.height)
		#make sure upper bound isn't too small for lower bound
		while x1 < 10 or y1 < 10:
			x1 = np.random.randint(self.width)
			y1 = np.random.randint(self.height)
		x0 = np.random.randint(x1)
		y0 = np.random.randint(y1)
		#rgb values for outline---------
		ro = np.random.randint(255)
		go = np.random.randint(255)
		bo = np.random.randint(255)
		a = np.random.randint(255)
		#rgb values for fill------------
		rf = np.random.randint(255)
		gf = np.random.randint(255)
		bf = np.random.randint(255)
		#-------------------------------	
		end = np.random.randint(360) #ending angle
		#make end angle isn't too small for start angle
		while end < 10:
			end = np.random.randint(360)
		start = np.random.randint(end) #starting angle
		lwidth = np.random.randint(10) #width of the line in pixels

		if shape == 0:
			self.drawing.arc([(x0,y0),(x1,y1)],start,end,fill=(ro,go,bo,a))	
		elif shape == 1:
			self.drawing.rectangle([(x0,y0),(x1,y1)],fill=(rf,gf,bf,a),outline=(ro,go,bo,a))
		elif shape == 2:
			self.drawing.ellipse([(x0,y0),(x1,y1)],fill=(rf,gf,bf,a),outline=(ro,go,bo,a))
		elif shape == 3:
			self.drawing.pieslice([(x0,y0),(x1,y1)],start,end,fill=(rf,gf,bf,a),outline=(ro,go,bo,a))
		elif shape == 4:
			self.drawing.chord([(x0,y0),(x1,y1)],start,end,fill=(rf,gf,bf,a),outline=(ro,go,bo,a))
		else:
			self.drawing.line([(x0,y0),(x1,y1)],fill = (ro,go,bo,a),width = lwidth)			
		
	def plot_pic(self):
		self.pic.show()
		
	def display_image(self):
		display(self.pic)

if __name__ == '__main__':
	rand_im = generate_rand_image(width = 512, height =288)
	for i in range(100):
		j = np.random.randint(6) #generate either 0, 1, 2, 3, 4, or 5
		rand_im.draw(j)
	rand_im.plot_pic()
	#rand_im.display_image()
	#tensor_playground = Image.open('spiral_tensorflow_playground.png')
	#display(tensor_playground)

