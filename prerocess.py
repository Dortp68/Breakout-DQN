from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces.box import Box
from gymnasium.core import Wrapper
import cv2
import numpy as np
import gymnasium as gym
import ale_py
import shimmy

class PreprocessAtari(ObservationWrapper):
    def __init__(self, env):
        ObservationWrapper.__init__(self, env)
        self.img_size = (84, 84)
        self.observation_space = Box(0.0, 1.0, (self.img_size[0], self.img_size[1], 1))

    def observation(self, img):
        img = img[30:-16, :]
        # resize image
        img = cv2.resize(img, self.img_size)
        return img


class FrameBuffer(Wrapper):
    def __init__(self, env, n_frames=4):
        super(FrameBuffer, self).__init__(env)
        height, width, n_channels = env.observation_space.shape
        obs_shape = [n_channels * n_frames, height, width]
        self.observation_space = Box(0, 255, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'uint8')

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self, self.env.reset()[0])
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns frame buffer"""
        new_img, reward, terminated, truncated, info = self.env.step(action)
        self.update_buffer(self, new_img)
        return self.framebuffer, reward, terminated, truncated, info

    @staticmethod
    def update_buffer(self, img):
        offset = self.env.observation_space.shape[-1]
        axis = 0
        cropped_framebuffer = self.framebuffer[offset:, :, :]
        self.framebuffer = np.concatenate([cropped_framebuffer, img[None, :, :]], axis=axis)