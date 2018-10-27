import numpy as np
import os
import gym
import time
from gym import error, spaces
from gym import utils
from gym.utils import seeding
from ple import PLE
from ple.games.monsterkong import MonsterKong

class MonsterKongEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, map_config):
        self.map_config = map_config
        self.game = MonsterKong(self.map_config)

        self.fps = 30
        self.frame_skip = 1
        self.num_steps = 1
        self.force_fps = True
        self.display_screen = True
        self.nb_frames = 500
        self.reward = 0.0
        self.episode_end_sleep = 0.2

        if map_config.has_key('fps'):
            self.fps = map_config['fps']
        if map_config.has_key('frame_skip'):
            self.frame_skip = map_config['frame_skip']
        if map_config.has_key('force_fps'):
            self.force_fps = map_config['force_fps']
        if map_config.has_key('display_screen'):
            self.display_screen = map_config['display_screen']
        if map_config.has_key('episode_length'):
            self.nb_frames = map_config['episode_length']
        if map_config.has_key('episode_end_sleep'):
            self.episode_end_sleep = map_config['episode_end_sleep']
        self.current_step = 0

        self._seed()

        self.p = PLE(self.game, fps=self.fps, frame_skip=self.frame_skip, num_steps=self.num_steps,
        force_fps=self.force_fps, display_screen=self.display_screen, rng=self.rng)

        self.p.init()

        self._action_set = self.p.getActionSet()[1:]
        self.action_space = spaces.Discrete(len(self._action_set))
        (screen_width, screen_height) = self.p.getScreenDims()
        self.observation_space = spaces.Box(low=0, high=255, shape=(screen_height, screen_width, 3))

    def _seed(self, seed=24):
        self.rng = seed

    def _step(self, action_taken):
        reward = 0.0
        action = self._action_set[action_taken]
        reward += self.p.act(action)
        obs = self.p.getScreenRGB()
        done = self.p.game_over()
        info = {'PLE': self.p}
        self.current_step += 1
        if self.current_step >= self.nb_frames:
            done = True
        return obs, reward, done, info

    def _reset(self):
        self.current_step = 0
        # Noop and reset if done
        start_done = True
        while start_done:
            self.p.reset_game()
            _, _, start_done, _ = self._step(4)
            #self.p.init()
        if self.p.display_screen:
            self._render()
            if self.episode_end_sleep > 0:
                time.sleep(self.episode_end_sleep)
        return self.p.getScreenRGB()

    def _render(self, mode='human', close=False):
        if close:
            return  # TODO: implement close
        original = self.p.display_screen
        self.p.display_screen = True
        self.p._draw_frame()
        self.p.display_screen = original