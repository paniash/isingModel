# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 22:58:15 2021

@author: gauta
"""

import imageio
import glob
from PIL import Image


# images = [cv2.imread(file) for file in glob.glob("*.png")]

# print(images[1])
# # for filename in images:
# #     images.append(imageio.imread(filename))
# imageio.mimsave('movie.gif', images)


def make_gif(frame_folder):
    
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.png")]
    print(len(frames))
    imageio.mimsave('ridge-graf.gif', frames, format='GIF', fps=20)

make_gif(r"C:\Users\gauta\Downloads\Ridge n variation")

