import sys
import os
import os.path as osp
import argparse
import gym
from gym import wrappers
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
from scipy.stats import mode
from agent import  NaiveAgent
from oodpmodel import *
from utils import *
from atari_wrappers import *
from gym.envs.registration import register
import imp

# model parameters
max_iter = 5*10**5
learning_rate = 1e-4
batch_size = 16
height = width = 80
channels = 3
object_maxnum = 4
horizon = 33
lambda_pro = 1
lambda_h = 1
lambda_p = 10
lambda_e = 1
lambda_r=0
lambda_c=0
lambda_b=0
respath='results/'

# generalization problem
train_envs=5
test_envs=10

if not os.path.exists(respath):
	os.makedirs(respath)

def wrap_env(env, seed):  # non-atari
	env.seed(seed)

	expt_dir = 'tmp/transfer-gym/'
	env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)

	return env

def output_acc(model,replay_buffer,test_buffer,bgimage):
	from evaluator import Evaluator
	evaluator=Evaluator()
	
	def calc_acc(buffer,is_train):
		obs, actions, rewards,nextobs, _,env_id,poss,poss_next = buffer.sample(batch_size)
		pred_obs = model.sess.run(model.Pred_nextI,
			feed_dict={
				model.input_I : obs,
				model.input_actions : actions,
				model.input_bgI :bgimage[env_id],
				model.input_nextI : nextobs,
				model.truepos:poss,
				model.truepos_next:poss_next,
				model.is_train:is_train
			})
		pred_obs=np.transpose((pred_obs/2.0+0.5)*255.0,[0,2,1,3])
		dis=np.zeros(shape=(7),dtype=np.int32)
		for i in range(batch_size):
			x,y=poss_next[i]
			dis+=evaluator.calc_dis(nextobs[i],pred_obs[i],x,y)
		return dis
	
	train_acc=np.zeros(shape=(7),dtype=np.float32)
	test_acc=np.zeros(shape=(7),dtype=np.float32)
	for i in range(100):
		train_acc+=calc_acc(replay_buffer,True)
		test_acc+=calc_acc(test_buffer,False)
	
	for i in range(1,7):
		train_acc[i]+=train_acc[i-1]
		test_acc[i]+=test_acc[i-1]
	train_acc = train_acc/train_acc[6]
	test_acc = test_acc/test_acc[6]

	print '========Training environments========'
	for i in range(3):
		print str(i)+'-error accuracy: '+str(train_acc[i])
	print '========Testing environments========'
	for i in range(3):
		print str(i)+'-error accuracy: '+str(test_acc[i])

def main():
	# Get arguments
	parser = argparse.ArgumentParser()
	parser.add_argument('-map_config', type=str,
						help='The map and config you want to run in MonsterKong.',default='configs/8blocks/level0')
	args = parser.parse_args()

	envlist=[]
	for i in range(train_envs+test_envs):
		try:
			map_config_file = args.map_config+'_'+str(i)+'.py'
			map_config = imp.load_source('map_config', map_config_file).map_config
		except Exception as e:
			sys.exit(str(e) + '\n'
				 +'map_config import error. File not exist or map_config not specified')

		register(
			id='MonsterKong-v'+str(i),
			entry_point='ple.gym_env.monsterkong:MonsterKongEnv',
			kwargs={'map_config': map_config},
		)
		env = gym.make('MonsterKong-v'+str(i))
		env = ProcessFrame(env)
		# Though the seed is unused here, it will be useful when incorapting an environment generator 
		# that can randomly generate reasonable environments.
		seed = np.random.randint(0, 1000) 
		env = wrap_env(env, seed)
		envlist.append(env)

	# other parameters
	test_collection_iter=10**4
	test_batch_num=100
	replay_buffer_size = max_iter
	learn_start = 10**4
	fore_max_iter = 10**3
	Dyn_threshold = 0.01
	movethreshold = 10000

	replay_buffer = SamplingReplayBuffer(replay_buffer_size, 1)
	test_buffer = ReplayBuffer(replay_buffer_size, 1)
	num_actions = env.action_space.n
	agent = NaiveAgent(num_actions)

	model=OODP(
		lr = learning_rate,
		BATCH_SIZE = batch_size,
		HEIGHT = height,
		WIDTH = width,
		CHANNELS = channels,
		object_maxnum = object_maxnum,
		horizon = horizon,
		action_dim = num_actions,
		Dyn_threshold = Dyn_threshold,
		lambda_pro = lambda_pro,
		lambda_h = lambda_h,
		lambda_p = lambda_p,
		lambda_e = lambda_e,
		lambda_r = lambda_r,
		lambda_c = lambda_c,
		lambda_b = lambda_b,
		path = respath,
		Isloadcheckpoint=False)

	
	#=========================== Test samples collection ===========================
	env_index = random.choice(range(train_envs,train_envs+test_envs))
	ob = envlist[env_index].reset()
	pos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)

	r = 0
	done = False
	# buffer (different from Q leanring)
	# current state, current reward, whether current state is done, current action, index of env
	for i in range(test_collection_iter):
		action=agent.pickAction(ob)

		idx = test_buffer.store_frame(ob)
		test_buffer.store_effect(idx, action, r, done,env_index,pos)

		ob, r, done, _ = envlist[env_index].step(action)
		pos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)


		if done:
			
			action=4
			idx = test_buffer.store_frame(ob)
			test_buffer.store_effect(idx, action, r, done,env_index,pos)
			env_index = random.choice(range(train_envs,train_envs+test_envs))
			ob = envlist[env_index].reset()
			r = 0
			done = False
			pos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)

	# make sure the last env is done
	while (not done):
		action=agent.pickAction(ob)
		_, _, done,_ = envlist[env_index].step(action)


	#=========================== Foreground detection ============================
	
	for i in range(train_envs+test_envs):
		done =False
		ob = envlist[i].reset()
		tmp=np.empty([fore_max_iter] + list(ob.shape), dtype=np.uint8)
		for j in range(fore_max_iter):
			action=agent.pickAction(ob)
			if done:
				ob = envlist[i].reset()
				done = False
			else:
				ob, r, done, _ = envlist[i].step(action)
			tmp[j]=ob
		if i==0:
			bgimage=np.empty([train_envs+test_envs] + list(ob.shape), dtype=np.uint8)
		tmp,_=mode(tmp)
		bgimage[i]=np.squeeze(tmp,0)

		# make sure the last env is done
		while (not done):
			action=agent.pickAction(ob)
			_, _, done,_ = envlist[i].step(action)
		

		done =False

	#=========================== Sample test data ============================
	test_obset=[]
	test_nextobset=[]
	test_actionset=[]
	test_bgimageset=[]
	test_poset=[]
	test_nextposet=[]
	for i in range(test_batch_num):
		test_obs, test_actions, test_rewards, test_nextob,_,test_env_id,test_pos,test_nextpos = test_buffer.sample(batch_size)
		test_obset.append(test_obs)
		test_actionset.append(test_actions)
		test_bgimageset.append(bgimage[test_env_id])
		test_nextobset.append(test_nextob)
		test_poset.append(test_pos)
		test_nextposet.append(test_nextpos)


	#=========================== Run training ============================
	# train one step, collect one sample (interact with the environment once, which is similar with DQN)
	env_index = random.choice(range(train_envs))
	ob = envlist[env_index].reset()
	tpos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)

	r = 0
	done = False
	# Buffer contains current state, current reward, whether current state is done, current action, index of env
	for i in range(max_iter):
		action=agent.pickAction(ob)

		idx = replay_buffer.store_frame(ob)
		replay_buffer.store_effect(idx, action, r, done,env_index,tpos)

		pob=ob
		ob, r, done, _ = envlist[env_index].step(action)
		tpos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)
		dx=AgentMotion(pob,ob,movethreshold)
		replay_buffer.store_dx(idx,dx)


		if done:
			
			action=4
			idx = replay_buffer.store_frame(ob)
			replay_buffer.store_effect(idx, action, r, done,env_index,tpos)
			replay_buffer.store_dx(idx,0)
			env_index = random.choice(range(train_envs))
			ob = envlist[env_index].reset()
			tpos=np.array(envlist[env_index].env.env.p.game.newGame.Players[0]._Person__position)
			r = 0
			done = False

		if (replay_buffer.can_sample(batch_size) and i>learn_start):
			obs, actions, rewards,nextobs, _,env_id,poss,poss_next = replay_buffer.sample(batch_size)
			model.update(obs,actions,nextobs,bgimage[env_id],\
				test_batch_num,test_obset,test_actionset,test_nextobset,test_bgimageset,poss,poss_next,test_poset,test_nextposet)

	output_acc(model,replay_buffer,test_buffer,bgimage)

if __name__ == "__main__":
	main()