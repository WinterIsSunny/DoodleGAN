#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 15:03:54 2018

@author: yusu
"""

from keras.layers import Input, Dense, Flatten, Dropout, Reshape, Concatenate
from keras.layers import BatchNormalization, Activation, Conv2D, Conv2DTranspose,UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras.optimizers import Adam
from keras.datasets import cifar10
import keras.backend as K
from keras.preprocessing import image

import matplotlib.pyplot as plt
import sys
import numpy as np
import os
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

os.makedirs('images', exist_ok=True)

matplotlib.interactive(True)

channels = 1
img_size = 28
img_w = img_h = img_size
img_shape = (img_size, img_size, channels)
n_epochs = 200

classes = ['saxophone',
        'raccoon',
        'piano',
        'panda',
        'leg',
        'headphones',
        'ceiling_fan',
        'bed',
        'basket',
        'aircraft_carrier']

#  Generator
def get_generator(input_layer, condition_layer):
    depth = 64
    p = 0.4
    merged_input = Concatenate()([input_layer, condition_layer])  

    dense1 = Dense(7*7*64)(merged_input)
    dense1 = BatchNormalization(axis=-1,momentum=0.9)(dense1)
    dense1 = Activation(activation='relu')(dense1)
    dense1 = Reshape((7,7,64))(dense1)
    dense1 = Dropout(p)(dense1)

    # Convolutional layers
    conv1 = UpSampling2D()(dense1)
    conv1 = Conv2DTranspose(int(depth/2), kernel_size=5, padding='same', activation=None,)(conv1)
    conv1 = BatchNormalization(axis=-1,momentum=0.9)(conv1)
    conv1 = Activation(activation='relu')(conv1)

    conv2 = UpSampling2D()(conv1)
    conv2 = Conv2DTranspose(int(depth/4), kernel_size=5, padding='same', activation=None,)(conv2)
    conv2 = BatchNormalization(axis=-1,momentum=0.9)(conv2)
    conv2 = Activation(activation='relu')(conv2)

    #conv3 = UpSampling2D()(conv2)
    conv3 = Conv2DTranspose(int(depth/8), kernel_size=5, padding='same', activation=None,)(conv2)
    conv3 = BatchNormalization(axis=-1,momentum=0.9)(conv3)
    conv3 = Activation(activation='relu')(conv3)

    # Define output layers
    output = Conv2D(1, kernel_size=5, strides=1, padding="same")(conv3)
    
    out = Activation("sigmoid")(output)
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()
  
    return model,out



# discriminator
def get_discriminator(input_layer, condition_layer,depth = 64,p = 0.4):
    
    #merged_input = Concatenate()([input_layer, condition_layer])  
    
    conv1 = Conv2D(depth*1, 5, strides=2, padding='same', activation='relu')(input_layer)
    conv1 = Dropout(p)(conv1)

    conv2 = Conv2D(depth*2, 5, strides=2, padding='same', activation='relu')(conv1)
    conv2 = Dropout(p)(conv2)

    conv3 = Conv2D(depth*4, 5, strides=2, padding='same', activation='relu')(conv2)
    conv3 = Dropout(p)(conv3)

    conv4 = Conv2D(depth*8, 5, strides=1, padding='same', activation='relu')(conv3)
    conv4 = Flatten()(Dropout(p)(conv4))
    
    merged_layer = Concatenate()([conv4, condition_layer])
    output = Dense(512, activation='relu')(merged_layer)
    #hid = Dropout(0.4)(hid)
    out = Dense(1, activation='sigmoid')(output)
    model = Model(inputs=[input_layer, condition_layer], outputs=out)
    model.summary()
    
    return model,out


def one_hot_encode(y):
    z = np.zeros((len(y), 10))
    idx = np.arange(len(y))
    #print(type(idx[0]))
    #for i in range(len(y)):
     #   z[i,y[i]] = 1
    z[idx,y] = 1
    return z

def generate_noise(n_samples, noise_dim):
    X = np.random.normal(0, 1, size=(n_samples, noise_dim))
    return X

def generate_random_labels(n):
    y = np.random.choice(10, n)
    y = one_hot_encode(y)
    #print(y.shape)
    return y

img_input = Input(shape=(28,28,1))
disc_condition_input = Input(shape=(10,))
discriminator, disc_out = get_discriminator(img_input, disc_condition_input)
discriminator.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy', metrics=['accuracy'])
#discriminator.trainable = False

noise_input = Input(shape=(100,))
gen_condition_input = Input(shape=(10,))
generator, gen_out = get_generator(noise_input, gen_condition_input)

gan_input = Input(shape=(100,))
x = generator([gan_input, gen_condition_input])
gan_out = discriminator([x, disc_condition_input])
AM = Model(inputs=[gan_input, gen_condition_input, disc_condition_input], output=gan_out)
AM.summary()
AM.compile(optimizer=Adam(0.0002, 0.5), loss='binary_crossentropy')

def get_all_classes():
    df = pd.DataFrame([], columns=['Image', 'Label'])
    for i, label in enumerate(classes):
        data = np.load('./data/%s.npy' % label) / 255
        data = np.reshape(data, [data.shape[0], img_size, img_size, 1])
        df2 = pd.DataFrame([(row, i) for row in data], columns=['Image', 'Label'])
        df = df.append(df2)
    return df.sample(frac=1) # shuffle

def save_model(model_json, name):
    with open(name, "w+") as json_file:
        json_file.write(model_json)

def save_real_imgs(real_imgs):
    doodle_per_img = 16
    for i in range(real_imgs.shape[0] - doodle_per_img):
        plt.figure(figsize=(5,5))
        for k in range(doodle_per_img):
            plt.subplot(4, 4, k+1)
            plt.imshow(real_imgs.iloc[i + k].reshape((img_size, img_size)), cmap='gray')
            plt.axis('off')
        print("Saving {}".format(i))
        plt.tight_layout()
        plt.show()
        plt.savefig('./images/real_{}.png'.format(i+1))
        
def make_trainable(net, is_trainable):
    net.trainable = is_trainable
    for l in net.layers:
        l.trainable = is_trainable
        
        
def train(df,epochs=20,batch=128):
    
    exp_replay = []
    num_batches = int(df.shape[0]/batch)
    labels = one_hot_encode(np.array(df['Label']).astype(np.int64))
    
    d_loss = []
    a_loss = []
    running_d_loss = 0
    running_d_acc = 0
    running_a_loss = 0
    running_a_acc = 0
    for i in range(1, epochs+1):
        for batch_idx in range(num_batches):
            images = np.array([np.reshape(row, (28, 28, 1)) for row in df['Image'].iloc[batch_idx*batch:(batch_idx+1)*batch]])
            y = labels[batch_idx*batch:(batch_idx+1)*batch] 

            noise_data = generate_noise(batch, 100)
            random_labels = generate_random_labels(batch)
            generated_images = generator.predict([noise_data,y])
            
            true_labels = np.zeros((batch, 1)) + np.random.uniform(low=0.0, high=0.1, size=(batch, 1))
            # Train discriminator on real data
            d_loss_true = discriminator.train_on_batch([images, y], true_labels)
            
            gene_labels = np.ones((batch, 1)) - np.random.uniform(low=0.0, high=0.1, size=(batch, 1))
            #train discriminator on generated data
            d_loss_gene = discriminator.train_on_batch([generated_images, y], gene_labels)
            
            # Store a random point for experience replay
            r_idx = np.random.randint(batch)
            exp_replay.append([generated_images[r_idx], y[r_idx], gene_labels[r_idx]])

            #If we have enough points, do experience replay
            if len(exp_replay) == batch:
                generated_images = np.array([p[0] for p in exp_replay])
                labels = np.array([p[1] for p in exp_replay])
                gene_labels = np.array([p[2] for p in exp_replay])
                expprep_loss_gene = discriminator.train_on_batch([generated_images, labels], gene_labels)
                exp_replay = []
                break
                
            d_loss.append(0.5 * np.add(d_loss_true, d_loss_gene))
            running_d_loss += d_loss[-1][0]
            running_d_acc += d_loss[-1][1]
            
            
            noise_data = generate_noise(batch, 100)
            random_labels = generate_random_labels(batch)
            a_loss.append(AM.train_on_batch([noise_data, random_labels, random_labels], np.zeros((batch, 1))))
            #running_a_loss += a_loss[-1][0]
            #running_a_acc += a_loss[-1][1]
            
            print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [D accu : %f]" % (i,epochs, batch_idx, num_batches,
                                                           running_d_loss , running_d_acc))
            #log_mesg = "%d: [D loss: %f, acc: %f]" % (i, running_d_loss/i, running_d_acc/i)
            #log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, running_a_loss/i, running_a_acc/i)
            #print(log_mesg)

        noise_data = generate_noise(batch, 100)
        random_labels = generate_random_labels(batch)
        # We use same labels for generated images as in the real training batch
        gen_imgs = generator.predict([noise_data, y])

        plt.figure(figsize=(5,5))
        for k in range(16):
            plt.subplot(4, 4, k+1)
            plt.imshow(gen_imgs[k, :, :, 0], cmap='gray')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
        plt.savefig('./images/{}.png'.format(i+1))

    return a_loss, d_loss


data = get_all_classes()

train(data, epochs=n_epochs, batch=128)


save_model(generator.to_json(), "generator.json")
save_model(AM.to_json(), "discriminator.json")

