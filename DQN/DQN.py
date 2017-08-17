
""" this is a work in progress.
Most of the code is taken from https://github.com/dennybritz/reinforcement-learning/blob/master/DQN/Deep%20Q%20Learning%20Solution.ipynb
"""



%matplotlib inline

import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import psutil
import tensorflow as tf

if "../" not in sys.path:
	sys.path.append("../")

from lib import plotting
from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")


class StateProcessor:

	def __init__(self):
		with tf.variable.scope("state_processor"):
			self.input_state = tf.placeholder(shape=[210,160,3], dtype = tf.unit8)
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
			self.output = tf.image.resoze_images(
				self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state):
		"""statae: a[210,160,3] whcih is an atari rgb state"""
		return sess.run(self.output, {self.input_state: state}) # sess.run will run
		#one step of tensor flow on the tensor output by substituting the values in the dict for the arguements 


class Estimator:
	"""Q - value estimator neural network.
	this network is used for both Q network and the target network

	"""

	def __init__(self, scope="estimator", summaries_dir=None):
		#placeholder for our input
		self.scope = scope
		self.summary_writer = None
		with tf.variable_scope(scope):
			self._build_model()

	def _build_model(self):
		#inputs are 4 rgb frames
		self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name='X')
		#td target
		self.y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
		#action selected
		self.actions_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="actions")
		
		X = tf.to_float(self.X_pl)/255.0
		batch_size = tf.shape(self.X_pl)[0]

		conv1 = tf.contrib.layers.conv2d(
			X, 32, 8, 4, activation_fn=tf.nn.relu)
		conv2 = tf.contrib.layers.conv2d(
			X, 64 4, 2, activation_fn=tf.nn.relu)
		conv3 = tf.contrib.layers.conv2d(
			X, 64, 4, 2, activation_fn=tf.nn.relu)

		#fully connected layers
		flattened = tf.contrib.layers.flatten(conv3)
		fcl = tf.contrib.layers.fully_connected(flattened, 512)
		self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

		gather_idx  = tf.range(batch_size)*tf.shape(self.predictions[1]+self.actions_pl)
		self.actions_predictions = tf.gather(tf.reshape(self.predictions, [-1], gather_idx))

		self.losses = tf.squared_difference(self.y_pl, self.actions_predictions)
		self.loss = tf.reduce_mean(self.losses)

		self.optimizer = tf.train.RMSPropOptimizer(0.00025, 0.99, 0.0, 1e-6)
		self.train_op = self.optimizer.minimize(self.loss, global_step=tf.contrib.framework.get_global_step())

		self.summaries = tf.summary.merge([
			tf.summary.scalar("loss", self.loss),
			tf.summary.histogram("loss_hist", self.losses),
			tf.summary.histogram("q_values_hist", self.predictions),
			tf.summary.scalar("max_q_values", tf.reduce_max(self.predictions))
			])

	def predict(self, sess, s):
			"""predict action values"""
		return sess.run(self.predictions, {self.X_pl:s})
	def update(self, sess, s, s. y):
		"""update the estimator towards the given target"""
		feed_dict = {self.X_pl:s, self.y_pl:y, self.actions_pl:a}
		summaries, global_step, _, loss = sess.run(
			[self.summaries, tf.contrib.framework.get_global_step(), self.train_op, self.loss], feed_dict)
		if self.summary_writer:
			self.summary_writer.add_summary(summaries, global_step)
		return loss


tf.reset_default_graph()
global_step = tf.Variable(0, name = "global_step", trainable=False)

e = Estimator(scope="test")
sp = StateProcessor()

with tf.Session as sess:
	sess.run(tf.global_variables_initializer())
	observation = env.reset()

	observation_p = sp.process(sess, observation)
	observation = np.stack([observation_p]*4, axis=2)
	observations = np.array([observation]*2)

	print(e.predict(sess, observations))

	y = np.array([10.0, 10.0])
	a = np.array([1,3])
	print(e.update(sess,observations,a,y))

class ModelParamaetersCopier():
	def __init__(self, estimator1, estimator2):
		e1_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator1.scope)]
		e1_params = sorted(e1_params, key=lambda v: v.name)
		e2_params = [t for t in tf.trainable_variables() if t.name.startswith(estimator2.scope)]
		e2 = sorted(e2_params, key=lambda v:v.name)
		self.update_ops = []
		for e1_v, e2_v in zip(e1_params, e2_params):
			op = e2_v.assign(e1_v)
			self.update_ops.append(op)

	def make(self, sess):
		sess.run(self.update_ops)

def make_epsilon_greedy_policy(estimator, nA):
	def policy_fn(sess, observation, epsilon):
		A = np.ones(nA, dtype=float)*epsilon/nA
		q_values = estimator.predict(sess, np.expand_dims(observation, 0))[0]
		best_Action = np.argmax(q_values)
		A[]best_Action += (1.0 - epsilon)
		return A
	return policy_fn

def deep_q_learning(sess,
	env, q_estimator,
	target_estimator,
	state_processor,
	num_episodes, experiment_dir,
	replay_memory_size=500000,
	replay_memory_init_size=500000,
	update_target_estimator_every=10000,
	discount_factor=0.99,
	epsilon_start=1.0,
	epsilon_end=0.1,
	epsilon_decay_steps=500000,
	batch_size=32,
	record_video_every=50):
	
	Transitions = namedtuple("Transition", ["state", "action", "reward","next_state","done"])

	#repaly memory
	replay_memory = []

	estimator_copy = ModelParamaetersCopier(q_estimator, target_estimator)

	#keep track of useful stats
	stats = plotting.EpisodeStats(
		episode_lengths=np.zeros(num_episodes),
		episode_rewards=np.zeros(num_episodes))





	#populate replay memory 
	state = env.reset()
	state = state_processor.process(sess, state)





tf.reset_default_graph()
experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))
global_step = tf.Variable(0, name='global_step', trainable=False)

q_estimator = Estimator(scope="q_estimator", summaries_dir=experiment_dir)
target_estimator = Estimator(scope="target_q")

state_processor = StateProcessor()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	for t , stats in deep_q_learning(sess,
		env,
		q_estimator=q_estimator,
		target_estimator=target_estimator,
		state_processor=state_processor,
		experiment_dir=experiment_dir,
		num_episodes=10000,
		replay_memory_size=5000000,
		replay_memory_init_size=50000,
		update_target_estimator_every=10000,
		epsilon_start=1.0,
		epsilon_end=0.1,
		epsilon_decay_steps=500000,
		discount_factor=0.99,
		batch_size=32
		):
		print("\nEpisode reawrd: {}".format(stats.episode_rewards[-1]))
















