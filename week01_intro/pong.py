"""Auxilary files for those who wanted to solve breakout with CEM or policy gradient"""
import gymnasium as gym
from gymnasium import Wrapper
from gymnasium.spaces import Box

import numpy as np
from PIL import Image


def make_pong():
    """creates breakout env with all preprocessing done for you"""
    return PreprocessAtari(gym.make("PongDeterministic-v4"))


class PreprocessAtari(Wrapper):
    def __init__(self, env, height=42, width=42,
                 crop=lambda img: img[34:34 + 160], n_frames=4):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.observation_space = Box(0.0, 1.0, [n_frames, height, width])
        self.framebuffer = np.zeros([n_frames, height, width])

    def reset(self):
        """resets breakout, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset()[0])
        return self.framebuffer

    def step(self, action):
        """plays breakout for 1 step, returns 4-frame buffer"""
        new_img, r, terminated, truncated, info = self.env.step(action)
        self.update_buffer(new_img)
        return self.framebuffer, r, terminated, truncated, info

    ###image processing###

    def update_buffer(self, img):
        img = self.preproc_image(img)
        self.framebuffer = np.vstack([img[None], self.framebuffer[:-1]])

    def preproc_image(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = np.array(Image.fromarray(img).resize(self.img_size)).mean(-1)
        img = img.astype('float32') / 255.
        return img
