import time

import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import xplot
import datetime
import os
import random

import matplotlib.pyplot as plt

def Scaling(input,factor):
	return tf.keras.backend.resize_images(input,factor,factor,"channels_first")

def Batchnorm(name, axes, inputs, is_train=tf.Variable(True, trainable=False), 
				decay = 0.9, epsilon = 0.00001, act = tf.identity):#NCHW format
	if ((axes == [0,2,3]) or (axes == [0,1,2])):
		if axes==[0,1,2]: #NHW
			inputs = tf.expand_dims(inputs, 1)
			#axes = [0,2,3]
			# Old (working but pretty slow) implementation:
			##########

			# inputs = tf.transpose(inputs, [0,2,3,1])

		# mean, var = tf.nn.moments(inputs, [0,1,2], keep_dims=False)
		# offset = lib.param(name+'.offset', np.zeros(mean.get_shape()[-1], dtype='float32'))
		# scale = lib.param(name+'.scale', np.ones(var.get_shape()[-1], dtype='float32'))
		# result = tf.nn.batch_normalization(inputs, mean, var, offset, scale, 1e-4)

		# return tf.transpose(result, [0,3,1,2])

		# New (super fast but untested) implementation:
		inputs = tf.transpose(inputs, [0, 2, 3, 1]) #NCHW -> NHWC


	x_shape = inputs.get_shape()
	params_shape = x_shape[-1:]


	from tensorflow.python.training import moving_averages
	from tensorflow.python.ops import control_flow_ops

	with tf.variable_scope(name) as vs:
		axis = list(range(len(x_shape) - 1))


		## 2.
		# if tf.__version__ > '0.12.1':
		#	 moving_mean_init = tf.zeros_initializer()
		# else:
		#	 moving_mean_init = tf.zeros_initializer

		offset = lib.param(name+'.offset', np.zeros(params_shape, dtype='float32'))
		scale = lib.param(name+'.scale', tf.random_normal(params_shape, mean=1.0, stddev =0.002  ))

		moving_mean = lib.param(name+'.moving_mean', np.zeros(params_shape, dtype='float32'), trainable=False)
		moving_variance = lib.param(name+'.moving_variance', np.ones(params_shape, dtype='float32'), trainable=False)


		## 3.
		# These ops will only be preformed when training.
		mean, variance = tf.nn.moments(inputs, axis)
		

		def mean_var_with_update():
			try:	# TF12
				update_moving_mean = moving_averages.assign_moving_average(
								moving_mean, mean, decay, zero_debias=False)	 # if zero_debias=True, has bias
				update_moving_variance = moving_averages.assign_moving_average(
								moving_variance, variance, decay, zero_debias=False) # if zero_debias=True, has bias
				# print("TF12 moving")
			except Exception as e:  # TF11
				update_moving_mean = moving_averages.assign_moving_average(
								moving_mean, mean, decay)
				update_moving_variance = moving_averages.assign_moving_average(
								moving_variance, variance, decay)
				# print("TF11 moving")
			with tf.control_dependencies([update_moving_mean, update_moving_variance]):
				return tf.identity(mean), tf.identity(variance)

		m, v = tf.cond(is_train, mean_var_with_update, lambda: (moving_mean, moving_variance))
		result = act( tf.nn.batch_normalization(inputs, m, v, offset, scale, epsilon))
		
		if ((axes == [0,2,3]) or (axes == [0,1,2])):
			result = tf.transpose(result, [0, 3, 1, 2]) #NHWC -> NCHW

		return result 

class OODP:

	def __init__(
		self,
		lr = 1e-5,
		beta1 = 0.5,
		beta2 = 0.9,
		BATCH_SIZE = 32,
		HEIGHT = 84,
		WIDTH = 84,
		CHANNELS = 3,
		object_maxnum = 6,
		horizon=32,
		action_dim = 5,
		Dyn_threshold = 0,
		lambda_pro = 1,
		lambda_h = 1,
		lambda_p = 1,
		lambda_e = 1,
		lambda_r = 1,
		lambda_c = 1,
		lambda_b = 1,
		path = 'try1/',
		LOG_EVERY_N_STEPS = 10**3,
		TEST_STEPS=10**3,
		SUMMARY_EVERY_N_STEPS = 10**3,
		Isloadcheckpoint = False
		):

		self.lr = lr
		self.beta1 = beta1 
		self.beta2 = beta2 
		self.BATCH_SIZE = BATCH_SIZE
		self.HEIGHT = HEIGHT
		self.WIDTH = WIDTH
		self.CHANNELS = CHANNELS

		self.object_maxnum = object_maxnum
		self.horizon = horizon
		self.action_dim = action_dim
		self.Dyn_threshold = Dyn_threshold
		self.lambda_pro = lambda_pro
		self.lambda_h = lambda_h
		self.lambda_p = lambda_p
		self.lambda_e = lambda_e
		self.lambda_r = lambda_r
		self.lambda_c = lambda_c
		self.lambda_b = lambda_b

		self.costs = []
		self.step = 0
		self.path = path
		self.log_time = LOG_EVERY_N_STEPS
		self.test_time = TEST_STEPS
		self.summary_time = SUMMARY_EVERY_N_STEPS

		self.create_network()		
		config = tf.ConfigProto()  
		config.gpu_options.allow_growth=True		
		self.sess = tf.Session(config=config)
		self.sess.run(tf.global_variables_initializer())

		self.saver = tf.train.Saver()
		if tf.gfile.Exists(self.path+'summary_train/'):
   			tf.gfile.DeleteRecursively(self.path+'summary_train/')
   		if tf.gfile.Exists(self.path+'summary_test/'):
   			tf.gfile.DeleteRecursively(self.path+'summary_test/')
		self.train_writer = tf.summary.FileWriter(self.path+'summary_train/', graph=self.sess.graph)
		self.test_writer = tf.summary.FileWriter(self.path+'summary_test/', graph=self.sess.graph)


		if Isloadcheckpoint:
			checkpoint = tf.train.get_checkpoint_state(self.path+'saved_networks/')
			if checkpoint and checkpoint.model_checkpoint_path:
				saver.restore(self.sess, checkpoint.model_checkpoint_path)
				print("Successfully loaded: ", checkpoint.model_checkpoint_path)
			else:
				print("Could not find old network weights")

	def SinglePerceptionNet(self,I,name,inchannel=3,is_train=None):

		# similar to the paper "learn object from pixels" [Under review as a conference paper at ICLR 2018]
		X1 = lib.ops.conv2d.Conv2D('OjbectDetector'+name+'.1', inchannel, 64, 3, I, stride=2)
		X1 = Batchnorm('OjbectDetector'+name+'.BN1', [0,2,3], X1,is_train)
		X1 = tf.nn.relu(X1)
		X2 = lib.ops.conv2d.Conv2D('OjbectDetector'+name+'.2', 64, 64, 3, X1, stride=2)
		X2 = Batchnorm('OjbectDetector'+name+'.BN2', [0,2,3], X2,is_train)
		X2 = tf.nn.relu(X2)
		X3 = lib.ops.conv2d.Conv2D('OjbectDetector'+name+'.3', 64, 64, 3, X2, stride=1)
		X3 = Batchnorm('OjbectDetector'+name+'.BN3', [0,2,3], X3,is_train)
		X3 = tf.nn.relu(X3)
		multiF = tf.concat([I, Scaling(X1,2), Scaling(X2,4), Scaling(X3,4)], 1)

		X4 = lib.ops.conv2d.Conv2D('OjbectDetector'+name+'.4', inchannel+64+64+64, 32, 1, multiF, stride=1)
		X4 = Batchnorm('OjbectDetector'+name+'.BN4', [0,2,3], X4,is_train)
		X4 = tf.nn.relu(X4)
		output = lib.ops.conv2d.Conv2D('OjbectDetector'+name+'.5', 32, 1, 3, X4, stride=1)
		output = Batchnorm('OjbectDetector'+name+'.BN5', [0,2,3], output,is_train)

		return output


	def OjbectDetector(self,I,inchannel=3,inheight=84,inwidth=84,object_maxnum=6,is_train=None,batch_size=16):
		# I : [batch_size, channel, height, width]
		# M_dynamic : [batch_size, height, width]
		# M_static : [batch_size, object_maxnum, height, width]

		M=self.SinglePerceptionNet(I,'a',inchannel,is_train)

		for i in range(object_maxnum):
			M= tf.concat([M,self.SinglePerceptionNet(I,'o'+str(i),inchannel,is_train)],1)

		M=tf.nn.softmax(M,1)
		M_dynamic = M[:,0,:,:]
		M_static = M[:,1:,:,:]

		EntropyLoss = -tf.reduce_mean(tf.reduce_sum(M[:batch_size,:,:,:]*tf.log(tf.clip_by_value(M[:batch_size,:,:,:],1e-10,1.0)),axis=1))
		return I,M_dynamic,M_static,EntropyLoss

	def CalCoordinates(self,dynamicmask,inheight,batch_size):
		# co_dynamic : [bacthsize,2]

		Pmap = tf.reduce_sum(dynamicmask, axis=[1, 2])
		Pmap = tf.clip_by_value(Pmap,1e-10,1e10)
		Xmap = tf.tile(tf.reshape(tf.range(inheight),[1,inheight,1]),[batch_size,1,inheight])
		Ymap = tf.tile(tf.reshape(tf.range(inheight),[1,1,inheight]),[batch_size,inheight,1])
		x_dynamic = tf.reduce_sum(dynamicmask*tf.cast(Xmap,tf.float32),axis=[1, 2])/Pmap
		y_dynamic = tf.reduce_sum(dynamicmask*tf.cast(Ymap,tf.float32),axis=[1, 2])/Pmap

		return tf.stack([x_dynamic,y_dynamic],1)


	def TailorModule(self,co_dynamic,M_dynamic,M_static,inheight,batch_size,horizon):
		
		# Mos : [batch_size, horizon, horizon, object_maxnum ]
		# Ma : [batch_size, horizon, horizon, 1 ]

		x0 = tf.cast(tf.floor(co_dynamic[:,0]), 'int32')
		x1 = x0 + 1
		y0 = tf.cast(tf.floor(co_dynamic[:,1]), 'int32')
		y1 = y0 + 1
				
		zero = tf.zeros([], dtype='int32')
		max_x=tf.cast(inheight - 1, 'int32')
		max_y=tf.cast(inheight - 1, 'int32')

		x0 = tf.clip_by_value(x0, zero, max_x)
		x1 = tf.clip_by_value(x1, zero, max_x)
		y0 = tf.clip_by_value(y0, zero, max_y)
		y1 = tf.clip_by_value(y1, zero, max_y)

		x0_f = tf.cast(x0, 'float32')
		x1_f = tf.cast(x1, 'float32')
		y0_f = tf.cast(y0, 'float32')
		y1_f = tf.cast(y1, 'float32')
		w1 = tf.reshape(((x1_f-co_dynamic[:,0]) * (y1_f-co_dynamic[:,1])), [-1,1,1,1])
		w2 = tf.reshape(((x1_f-co_dynamic[:,0]) * (co_dynamic[:,1]-y0_f)), [-1,1,1,1])
		w3 = tf.reshape(((co_dynamic[:,0]-x0_f) * (y1_f-co_dynamic[:,1])), [-1,1,1,1])
		w4 = tf.reshape(((co_dynamic[:,0]-x0_f) * (co_dynamic[:,1]-y0_f)), [-1,1,1,1])

		Mo1,Ma1 = self.Cropping(x0,y0,M_dynamic,M_static,batch_size,horizon)
		Mo2,Ma2 = self.Cropping(x0,y1,M_dynamic,M_static,batch_size,horizon)
		Mo3,Ma3 = self.Cropping(x1,y0,M_dynamic,M_static,batch_size,horizon)
		Mo4,Ma4 = self.Cropping(x1,y1,M_dynamic,M_static,batch_size,horizon)
		Mos = tf.add_n([tf.stop_gradient(w1)*Mo1, tf.stop_gradient(w2)*Mo2, tf.stop_gradient(w3)*Mo3, tf.stop_gradient(w4)*Mo4])
		Ma = tf.add_n([w1*Ma1, w2*Ma2, w3*Ma3, w4*Ma4])
		Ma = tf.stop_gradient(Ma)

		return Mos,Ma

	def Cropping(self,cx_a,cy_a,M_dynamic,M_static,batch_size,horizon):
		
		padMa = tf.pad(M_dynamic,[[0,0],[horizon/2,horizon/2],[horizon/2,horizon/2]],"CONSTANT")
		batch_idx = tf.tile(tf.reshape(tf.range(0, batch_size), [batch_size, 1, 1]), [1, horizon, horizon])
		x_idx = tf.tile(tf.reshape(tf.range(0, horizon), [1,  horizon, 1]), [batch_size, 1, horizon])+tf.reshape(cx_a,[-1,1,1])
		y_idx = tf.tile(tf.reshape(tf.range(0, horizon), [1, 1, horizon]), [batch_size, horizon, 1])+tf.reshape(cy_a,[-1,1,1])
		
		# objectes interaction area Mo_ia:[batch_size, horizon, horizon, object_maxnum ]
		# dynamic interaction area Ma_ia:[batch_size, horizon, horizon,1]
		indices = tf.stack([batch_idx, x_idx, y_idx], 3)

		padMo = tf.pad(M_static,[[0,0],[0,0],[horizon/2,horizon/2],[horizon/2,horizon/2]],"CONSTANT")
		padMo_tr=tf.transpose(padMo, [0, 2, 3, 1])
		Mo_ia = tf.gather_nd(padMo_tr, indices)
		Ma_ia = tf.gather_nd(padMa, indices)
		
		return Mo_ia,tf.expand_dims(Ma_ia,3)

	def	STNmodule(self,dco,I,M_dynamic,inheight,batch_size):

		# dco dx,dy: [batch_size,2]
		# x : max_x-dx
		# y : max_y-dy

		x=inheight - 1-dco[:,0]
		y=inheight - 1-dco[:,1]

		x0 = tf.cast(tf.floor(x), 'int32')
		x1 = x0 + 1
		y0 = tf.cast(tf.floor(y), 'int32')
		y1 = y0 + 1

		zero = tf.zeros([], dtype='int32')
		max_x=tf.cast(inheight - 1, 'int32')
		max_y=tf.cast(inheight - 1, 'int32')

		x0 = tf.clip_by_value(x0, zero, 2*max_x)
		x1 = tf.clip_by_value(x1, zero, 2*max_x)
		y0 = tf.clip_by_value(y0, zero, 2*max_y)
		y1 = tf.clip_by_value(y1, zero, 2*max_y)

		x0_f = tf.cast(x0, 'float32')
		x1_f = tf.cast(x1, 'float32')
		y0_f = tf.cast(y0, 'float32')
		y1_f = tf.cast(y1, 'float32')
		w1 = tf.reshape(((x1_f-x) * (y1_f-y)), [-1,1,1,1])
		w2 = tf.reshape(((x1_f-x) * (y-y0_f)), [-1,1,1,1])
		w3 = tf.reshape(((x-x0_f) * (y1_f-y)), [-1,1,1,1])
		w4 = tf.reshape(((x-x0_f) * (y-y0_f)), [-1,1,1,1])

		Pred_Ia1,Pred_dynamicmask1 = self.CroppingForPred(x0-max_x,y0-max_y,I,M_dynamic,batch_size,inheight)
		Pred_Ia2,Pred_dynamicmask2 = self.CroppingForPred(x0-max_x,y1-max_y,I,M_dynamic,batch_size,inheight)
		Pred_Ia3,Pred_dynamicmask3 = self.CroppingForPred(x1-max_x,y0-max_y,I,M_dynamic,batch_size,inheight)
		Pred_Ia4,Pred_dynamicmask4 = self.CroppingForPred(x1-max_x,y1-max_y,I,M_dynamic,batch_size,inheight)
		Pred_Ia = tf.add_n([w1*Pred_Ia1, w2*Pred_Ia2, w3*Pred_Ia3, w4*Pred_Ia4])
		Pred_dynamicmask= tf.add_n([w1*Pred_dynamicmask1, w2*Pred_dynamicmask2, w3*Pred_dynamicmask3, w4*Pred_dynamicmask4])

		return Pred_Ia,Pred_dynamicmask

	def CroppingForPred(self,ndx,ndy,I,M_dynamic,batch_size,inheight):
		
		# ndx, ndy = -dx, -dy : [batch_size]
		# I : [batch_size, channel, height, width]
		# M_dynamic : [batch_size, height, width]

		padI = tf.pad(I,[[0,0],[0,0],[inheight,inheight],[inheight,inheight]],"CONSTANT",constant_values=-1.0)
		padMa= tf.pad(M_dynamic,[[0,0],[inheight,inheight],[inheight,inheight]],"CONSTANT",constant_values=0.0)

		batch_idx = tf.tile(tf.reshape(tf.range(0, batch_size), [batch_size, 1, 1]), [1, inheight, inheight])
		# -dx+|dx|
		x_idx = tf.tile(tf.reshape(tf.range(0, inheight), [1,  inheight, 1]), [batch_size, 1, inheight])+tf.reshape(ndx+inheight,[-1,1,1])
		y_idx = tf.tile(tf.reshape(tf.range(0, inheight), [1, 1, inheight]), [batch_size, inheight, 1])+tf.reshape(ndy+inheight,[-1,1,1])
		
		# Pred_Ia : [batch_size, horizon, horizon, 3 ]
		# Pred_dynamicmask : [batch_size, horizon, horizon,1]
		indices = tf.stack([batch_idx, x_idx, y_idx], 3)

		padI_tr=tf.transpose(padI, [0, 2, 3, 1])
		Pred_Ia = tf.gather_nd(padI_tr, indices)

		indices = tf.stack([batch_idx, x_idx, y_idx], 3)
		Pred_dynamicmask = tf.gather_nd(padMa, indices)
		
		return Pred_Ia,tf.expand_dims(Pred_dynamicmask,3)

	def RelationNet(self,Mo,Ma,actions,name,horizon,batch_size,action_dim,is_train=None):
		# Mo : [batch_size, horizon, horizon, 1 ]
		# actions : [batch_size,action_dim]
		# RlNout : [batch_size,4*4*32]
		# MoNout : [batch_size,1,2]

		# ==============Relation Net==============
		reMo=tf.reshape(Mo,[-1,1,horizon,horizon])
		x_idxes =tf.cast(tf.tile(tf.reshape(tf.range(-(horizon//2), horizon//2+1), \
			[1, 1, horizon, 1]), [batch_size, 1, 1, horizon]),tf.float32)
		y_idxes = tf.cast(tf.tile(tf.reshape(tf.range(-(horizon//2), horizon//2+1), \
		 [1, 1, 1, horizon]), [batch_size, 1, horizon, 1]),tf.float32)
		# [batch_size*object_maxnum, 2 ,horizon,horizon ]
		RlNout = lib.ops.conv2d.Conv2D('RCN'+name+'.1', 3, 16, 3, tf.concat([reMo,x_idxes,y_idxes],1), stride=2)
		RlNout = Batchnorm('RCN'+name+'.BN1', [0,2,3], RlNout,is_train)
		RlNout = tf.nn.relu(RlNout)
		RlNout = lib.ops.conv2d.Conv2D('RCN'+name+'.2', 16, 32, 3, RlNout, stride=2)
		RlNout = Batchnorm('RCN'+name+'.BN2', [0,2,3], RlNout,is_train)
		RlNout = tf.nn.relu(RlNout)
		RlNout = lib.ops.conv2d.Conv2D('RCN'+name+'.3', 32, 64, 3, RlNout, stride=2)
		RlNout = Batchnorm('RCN'+name+'.BN3', [0,2,3], RlNout,is_train)
		RlNout = tf.nn.relu(RlNout)
		RlNout = lib.ops.conv2d.Conv2D('RCN'+name+'.4', 64, 128, 3, RlNout, stride=2)
		RlNout = Batchnorm('RCN'+name+'.BN4', [0,2,3], RlNout,is_train)
		RlNout = tf.nn.relu(RlNout)
		RlNout = tf.reshape(RlNout, [batch_size, -1])

		MoNout = lib.ops.linear.Linear('RCN'+name+'.Mo1', 3*3*128, 64, RlNout)
		MoNout = tf.nn.relu(MoNout)
		MoNout = lib.ops.linear.Linear('RCN'+name+'.Mo2', 64, 2*action_dim, MoNout)

		return tf.expand_dims(MoNout,1)

	def DynamicsNet(self,Mos,Ma,actions,batch_size,horizon,action_dim,object_maxnum,is_train=None):
		# Mos : [batch_size, horizon, horizon, object_maxnum ]
		# actions : [batch_size,action_dim]

		MoNout=self.RelationNet(Mos[:,:,:,0],Ma,actions,'o0',horizon,batch_size,action_dim,is_train)

		for i in range(1,object_maxnum):
			tmp1=self.RelationNet(Mos[:,:,:,i],Ma,actions,'o'+str(i),horizon,batch_size,action_dim,is_train)
			MoNout=tf.concat([MoNout,tmp1],1)

		# BkaNout : [batch_size,2]
		BkaNout = tf.Variable(tf.zeros([2*action_dim]))

		# tmp : [batch_size,2*action_dim]
		tmp = tf.reduce_sum(MoNout,1)+tf.expand_dims(BkaNout,0)
		# Pred_delt : [batch_size,2]
		Pred_delt=tf.stack([tf.reduce_sum(tmp[:,:action_dim]*actions,1),tf.reduce_sum(tmp[:,action_dim:]*actions,1)],1)
		
		return Pred_delt

	def BgNet(self,I,inchannel,batch_size,is_train=None):
		Conv1 = lib.ops.conv2d.Conv2D('BgNet.1', inchannel, 64, 3, I, stride=2)
		Conv1 = tf.nn.relu(Conv1)

		Conv2 = lib.ops.conv2d.Conv2D('BgNet.2', 64, 64, 3, Conv1, stride=2)
		Conv2 = tf.nn.relu(Conv2)

		Conv3 = lib.ops.conv2d.Conv2D('BgNet.3', 64, 64, 3, Conv2, stride=2)
		Conv3 = tf.nn.relu(Conv3)

		FC = tf.reshape(Conv3, [batch_size, 10*10*64])
		FC  = lib.ops.linear.Linear('BgNet.4', 10*10*64, 128, FC)
		FC = tf.nn.relu(FC)

		FC  = lib.ops.linear.Linear('BgNet.5', 128, 10*10*64, FC)
		FC = tf.nn.relu(FC)

		FC = tf.reshape(FC, [batch_size, 64,10,10])

		De1 = lib.ops.deconv2d.Deconv2D('BgNet.6', 64, 64, 3, FC)
		De1 = tf.nn.relu(De1)

		De2 = lib.ops.deconv2d.Deconv2D('BgNet.7', 64, 64, 3, De1)
		De2 = tf.nn.relu(De2)

		De3 = lib.ops.deconv2d.Deconv2D('BgNet.8', 64, 3, 3,De2)
		De3 = tf.tanh(De3)
		return De3

	def PredictLoss(self,groundtruth_dx,I,nextI,Mos,Ma,M_dynamic,M_dynamic_next,ImageBg,ImageBg_next,co_dynamic,co_dynamic_next,actions,\
		batch_size,action_dim,object_maxnum,inheight,horizon,is_train=None):

		# actions : [batch_size,action_dim]
		# co_dynamic : [batch_size,2]

		Pred_delt = self.DynamicsNet(Mos,Ma,actions,batch_size,horizon,action_dim,object_maxnum,is_train)

		
		pred_co = Pred_delt+co_dynamic

		Pred_Ia,Pred_M_dynamic=self.STNmodule(Pred_delt,I,M_dynamic,inheight,batch_size)

		#[batch_size, 3, horizon, horizon]
		M_dynamic=tf.expand_dims(M_dynamic,1)
		M_dynamic_next=tf.expand_dims(M_dynamic_next,1)
		Recon_I=(1-M_dynamic)*ImageBg+M_dynamic*I
		Recon_nextI=(1-M_dynamic_next)*ImageBg_next+M_dynamic_next*nextI

		BgLoss=tf.reduce_mean(tf.square(ImageBg-ImageBg_next))
		ConsistLoss=tf.reduce_mean(tf.square(tf.transpose(Pred_M_dynamic,[0,3,1,2])-M_dynamic_next))
		ReconLoss=0.5*tf.reduce_mean(tf.square(Recon_I-I))+0.5*tf.reduce_mean(tf.square(Recon_nextI-nextI))

		Dyn_pred = Pred_M_dynamic*Pred_Ia
		#[batch_size, horizon, horizon, 3]
		self.Pred_nextI=(1-Pred_M_dynamic)*tf.transpose(ImageBg,[0,2,3,1])+Pred_M_dynamic*Pred_Ia
		PredictionLoss = tf.reduce_mean(tf.square(self.Pred_nextI-tf.transpose(nextI,[0,2,3,1])))
		HighwayLoss = tf.reduce_sum(tf.square(co_dynamic_next - pred_co))/(2*batch_size)
		GTMotionLoss = tf.reduce_sum(tf.square(groundtruth_dx - Pred_delt))/(2*batch_size)

		return PredictionLoss,HighwayLoss,BgLoss,ConsistLoss,ReconLoss,GTMotionLoss,Dyn_pred
	
	def ProposalLoss(self,I,M_dynamic,dynamicproposal,bgI,inheight):

		# Weighted l2 loss
		
		weight=tf.reduce_sum(dynamicproposal,axis=[1,2],keep_dims=True)/(inheight*inheight)
		weightmask=dynamicproposal*(1-weight)+(1-dynamicproposal)*weight

		DALoss = tf.reduce_mean(tf.reduce_sum(tf.square(M_dynamic-dynamicproposal)*weightmask,axis=[1,2]))

		return DALoss
	
	def create_network(self):
		#=======Construct graph=======
		
		# input image : int8, 0-255, [batch_size, width, height, 3]
		self.is_train = tf.placeholder(tf.bool)
		self.input_I = tf.placeholder(tf.uint8, shape=[self.BATCH_SIZE, self.WIDTH, self.HEIGHT, self.CHANNELS])
		self.input_nextI = tf.placeholder(tf.uint8, shape=[self.BATCH_SIZE, self.WIDTH, self.HEIGHT, self.CHANNELS])
		self.input_bgI = tf.placeholder(tf.uint8, shape=[self.BATCH_SIZE,self.WIDTH, self.HEIGHT, self.CHANNELS])

		self.truepos = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,2])
		self.truepos_next = tf.placeholder(tf.float32, shape=[self.BATCH_SIZE,2])


		I = 2*((tf.cast(self.input_I, tf.float32)/255.)-.5)
		nextI = 2*((tf.cast(self.input_nextI, tf.float32)/255.)-.5)
		bgI = 2*((tf.cast(self.input_bgI, tf.float32)/255.)-.5)
		#	I:[batch_size, width, height, channel] --> [batch_size, channel, height, width]
		I = tf.transpose(I, [0, 3, 2, 1])
		nextI = tf.transpose(nextI, [0, 3, 2, 1])
		bgI = tf.transpose(bgI, [0, 3, 2, 1])
		
		# input image : int32, [batch_size]
		self.input_actions = tf.placeholder(tf.int32, shape=[self.BATCH_SIZE])
		actions = tf.cast(tf.one_hot(self.input_actions,self.action_dim),tf.float32)

		tmp=self.BgNet(tf.concat([I,nextI],0),self.CHANNELS,2*self.BATCH_SIZE,self.is_train)
		ImageBg=tmp[:self.BATCH_SIZE,:,:,:]
		ImageBg_next=tmp[self.BATCH_SIZE:,:,:,:]

		multiF,tmp,M_static,EntropyLoss = self.OjbectDetector(tf.concat([I,nextI],0),inchannel=self.CHANNELS,inheight=self.HEIGHT,inwidth=self.WIDTH,\
			object_maxnum=self.object_maxnum,is_train=self.is_train,batch_size=self.BATCH_SIZE)

		M_dynamic=tmp[:self.BATCH_SIZE,:,:]
		M_dynamic_next=tmp[self.BATCH_SIZE:,:,:]
		multiF=multiF[:self.BATCH_SIZE,:,:,:]
		M_static=M_static[:self.BATCH_SIZE,:,:,:]

		dynamicproposal=tf.reduce_mean(tf.square(I-bgI),axis=1)>=self.Dyn_threshold
		dynamicproposal=tf.cast(dynamicproposal, tf.float32)

		co_dynamic=self.CalCoordinates(M_dynamic,inheight=self.HEIGHT,batch_size=self.BATCH_SIZE)
		co_dynamic_next=self.CalCoordinates(M_dynamic_next,inheight=self.HEIGHT,batch_size=self.BATCH_SIZE)

		groundtruth_dx =(self.truepos_next-self.truepos)*((self.HEIGHT-1)/(120.0-1))
		groundtruth_dx=tf.stack([groundtruth_dx[:,1],groundtruth_dx[:,0]],1)

		Mos,Ma = self.TailorModule(co_dynamic,M_dynamic,M_static,inheight=self.HEIGHT,batch_size=self.BATCH_SIZE,horizon=self.horizon)

		ProLoss= self.ProposalLoss(I,M_dynamic,dynamicproposal,bgI=bgI,inheight=self.HEIGHT)
		PredictionLoss,HighwayLoss,BgLoss,ConsistLoss,ReconLoss,self.GTMotionLoss,Dyn_pred\
			= self.PredictLoss(groundtruth_dx,I,nextI,Mos,Ma,M_dynamic,M_dynamic_next,ImageBg,ImageBg_next,co_dynamic,co_dynamic_next,actions,\
				batch_size=self.BATCH_SIZE,action_dim=self.action_dim,object_maxnum=self.object_maxnum,\
			inheight=self.HEIGHT,horizon=self.horizon,is_train=self.is_train)

		self.Total_loss = self.lambda_pro * ProLoss +  self.lambda_h * HighwayLoss + self.lambda_p * PredictionLoss + \
					self.lambda_e * EntropyLoss + self.lambda_r * ReconLoss + self.lambda_c * ConsistLoss + self.lambda_b * BgLoss

		self.train_total_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.Total_loss)

		tf.summary.scalar('groundtruth_MotionLoss', self.GTMotionLoss)
		# The loss function is only used for the training process and makes no sence in unseen environments.
		tf.summary.scalar('TotalLoss', self.Total_loss*tf.cast(self.is_train,tf.float32))

		tf.summary.image('Image',tf.transpose(I,[0,2,3,1]) )
		tf.summary.image('Image_next',tf.transpose(nextI,[0,2,3,1]) )
		tf.summary.image('Prediction_of_Dynamic_Object',Dyn_pred)

		tf.summary.image('Dynamic_Object_Mask',tf.cast(tf.expand_dims(M_dynamic,3)>=0.5,tf.float32)*tf.transpose(I,[0,2,3,1]))
		for i in range(self.object_maxnum):
			tmp=tf.expand_dims(M_static[:,i,:,:],3)
			tf.summary.image('Static_Object_Mask_'+str(i),tf.cast(tmp>=0.5,tf.float32)*tf.transpose(I,[0,2,3,1]))

		self.summary_op = tf.summary.merge_all()  


	def update(self,I,actions,nextI,bgimage,test_batch_num,test_obset,test_actionset,test_nextobset,test_bgimageset,\
		poss,poss_next,test_poset,test_nextposet):


		_,cost = self.sess.run([self.train_total_op,self.Total_loss],
			feed_dict={
				self.input_I : I,
				self.input_actions : actions,
				self.input_bgI :bgimage,
				self.input_nextI : nextI,
				self.truepos:poss,
				self.truepos_next:poss_next,
				self.is_train:True
			})

		self.step += 1
		
		if self.step % self.log_time == 0 :
			self.saver.save(self.sess, self.path+'saved_networks/', global_step=self.step)

		if self.step % self.summary_time == 0:
			summary_str = self.sess.run(self.summary_op,feed_dict={
				self.input_I : I,
				self.input_actions : actions,
				self.input_bgI :bgimage,
				self.input_nextI : nextI,
				self.truepos:poss,
				self.truepos_next:poss_next,
				self.is_train:True
			})
			self.train_writer.add_summary(summary_str, self.step)
			self.train_writer.flush()

		if self.step % self.test_time == 0:	
			test_closses=[]

			for i in range(test_batch_num):
				test_closs = self.sess.run([self.GTMotionLoss],feed_dict={
					self.input_I : test_obset[i],
					self.input_actions : test_actionset[i],
					self.input_bgI :test_bgimageset[i],
					self.input_nextI : test_nextobset[i],
					self.truepos:test_poset[i],
					self.truepos_next:test_nextposet[i],
					self.is_train:False
				})
				test_closses.append(test_closs)

			summary_str = self.sess.run(self.summary_op,feed_dict={
					self.input_I : test_obset[i],
					self.input_actions : test_actionset[i],
					self.input_bgI :test_bgimageset[i],
					self.input_nextI : test_nextobset[i],
					self.truepos:test_poset[i],
					self.truepos_next:test_nextposet[i],
					self.is_train:False
				})	


			self.test_writer.add_summary(summary_str, self.step)
			self.test_writer.flush()

			train_ccost = self.sess.run([self.GTMotionLoss],
			feed_dict={
				self.input_I : I,
				self.input_actions : actions,
				self.input_bgI :bgimage,
				self.input_nextI : nextI,
				self.truepos:poss,
				self.truepos_next:poss_next,
				self.is_train:True
			})

			xplot.plot('Cost', cost)
			xplot.plot('groundtruth_MotionLoss_train', train_ccost)
			xplot.plot('groundtruth_MotionLoss_test', np.mean(test_closses))


		if (self.step < 5) or (self.step % self.test_time == self.test_time-1):
			xplot.flush(self.path)

		xplot.tick()