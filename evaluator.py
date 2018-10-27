import tensorflow as tf
import numpy as np
import cv2
from scipy.misc import imsave
import os
import shutil

def resize(x,y,shape1,shape2):
	w1,h1=shape1
	w2,h2=shape2
	x=x*w2/w1
	y=y*h2/h1
	return int(x),int(y)

class Evaluator():
	def __init__(self):
		self.create_conv()
	
	def create_conv(self):
		self.a = tf.placeholder(tf.float32, shape=(None,None,None,3))
		self.b = tf.placeholder(tf.float32, shape=(None,None,None,1))

		self.xy = tf.nn.depthwise_conv2d(self.a, self.b, (1,1,1,1), 'VALID')
		self.x2 = tf.nn.depthwise_conv2d(self.a**2, (self.b*0) + 1, (1,1,1,1), 'VALID')
		self.y2 = tf.reduce_sum(self.b**2, keep_dims=True)
		self.ans = self.x2 + self.y2 - 2*self.xy
		self.res = tf.reduce_sum(self.ans, axis=3)[0]
		
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		self.sess=tf.Session(config=config)
		
	def L2lossXY(self, A, B, xy, width=5, height=5):
		x, y = xy
		filters = np.zeros((width*2,height*2,3))
		a, c, b, d = x - width, x + width, y - height, y + height
		na, nc, nb, nd = max(a, 0), min(c, A.shape[0]), max(b, 0), min(d, A.shape[1])
		filters[na-a:nc-a,nb-b:nd-b] = A[na:nc,nb:nd]

		move_scale = 15
		inputs = np.zeros(((width+max(move_scale,width))*2, (height+max(move_scale,height))*2,3))
		I_std = np.zeros(((width+max(move_scale,width))*2, (height+max(move_scale,height))*2,3))
		a, c, b, d = x - width - max(move_scale,width), x + width + max(move_scale,width), y - height - max(move_scale,height), y + height + max(move_scale,height)
		na, nc, nb, nd = max(a, 0), min(c, A.shape[0]), max(b, 0), min(d, A.shape[1])
		inputs[na-a:nc-a,nb-b:nd-b] = B[na:nc,nb:nd]
		I_std[na-a:nc-a,nb-b:nd-b] = A[na:nc,nb:nd]
		
		dists = self.sess.run(self.res, feed_dict={self.a:[inputs], self.b:filters[:,:,:,None]})
		loc = dists.argmin()
		locy = loc % dists.shape[1]
		locx = loc // dists.shape[1]
		
		addx, addy = locx-max(move_scale,width), locy-max(move_scale,height)
		
		return addx, addy
		
	def calc_dis(self,ob_std,ob_predict,x,y):
		dis=np.zeros(shape=(7),dtype=np.int32)
		
		x,y=resize(x+2,y+2,(120.0,120.0),(80.0,80.0))
		dx,dy=resize(10,10,(120.0,120.0),(80.0,80.0))
		lx,ly=self.L2lossXY(ob_std,ob_predict,(x,y),dx,dy)
		dis[min(6,abs(lx)+abs(ly))]+=1
		
		return dis