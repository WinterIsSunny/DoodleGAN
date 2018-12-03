#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 22:48:00 2018

@author: yusu
"""
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import os
from matplotlib import pyplot as plt

#generator = load_model('generator.h5')
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
generator = model_from_json(loaded_model_json)
# load weights into new model
generator.load_weights("generator.h5")


for i in range(500):
    noise = np.random.uniform(-1.0, 1.0, size=[16, 100])
    gen_imgs = generator.predict(noise)
    plt.figure(figsize=(5,5))
    for k in range(gen_imgs.shape[0]):
        plt.subplot(4, 4, k+1)
        plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig('./images/image_{}.png'.format(i+1))