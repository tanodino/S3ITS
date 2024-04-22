# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:44:56 2021

@author: emmanuel
"""

# importation des modules nécessaires
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer, Dense, Activation, Dropout

tf.keras.backend.set_floatx('float32')



class Conv1D_bloc(Layer):
  def __init__(self, filters_nb, kernel_size, drop_val, **kwargs):
    super(Conv1D_bloc, self).__init__(**kwargs)
     
    self.conv1D = layers.Conv1D(filters_nb, kernel_size, padding="same",  kernel_initializer='he_normal')
    self.act = Activation('relu')
    self.output_ = Dropout(drop_val)
        
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    conv1D = self.conv1D(inputs)
    act = self.act(conv1D)
    return self.output_(act, training=training)

class TempCNN_Encoder(Layer):
  def __init__(self, flatten=True, drop_val=0.5, **kwargs):
    super(TempCNN_Encoder, self).__init__(**kwargs) # Appel du constructeur parent
     
    self.conv_bloc1 = Conv1D_bloc(64, 5, drop_val)
    self.conv_bloc2 = Conv1D_bloc(64, 5, drop_val)
    self.conv_bloc3 = Conv1D_bloc(64, 5, drop_val)
    
    self.flatten = None

    if flatten:
      self.flatten = layers.Flatten()
    else:
      self.flatten = layers.GlobalMaxPooling1D()
    
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    
    conv1 = self.conv_bloc1(inputs, training=training)
    conv2 = self.conv_bloc2(conv1, training=training)
    conv3 = self.conv_bloc3(conv2, training=training)

    flatten = self.flatten(conv3)
        
    return flatten


class Reconstruction(Layer):
  def __init__(self, nb_units, **kwargs):
    super(Reconstruction, self).__init__(**kwargs) # Appel du constructeur parent     
    self.dense = Dense(nb_units, activation=None)
  
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):    
    return self.dense(inputs)

class Classifier(Layer):
  def __init__(self, nb_class, nb_units, drop_val=0.5, **kwargs):
    super(Classifier, self).__init__(**kwargs) # Appel du constructeur parent
     
    self.dense = Dense(nb_units, activation='relu')     
    self.output_ = Dense(nb_class, activation="softmax")
  
  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):    
    dense = self.dense(inputs)
    return self.output_(dense)


class TempCNN_Model(keras.Model):
  def __init__(self, nb_class, nUnit2Reco, flatten=True, drop_val=0.5, **kwargs):
    super(TempCNN_Model, self).__init__(**kwargs) # Appel du constructeur parent

    # Feature Extractor
    self.encoder = TempCNN_Encoder(drop_val=drop_val, flatten=flatten)
    
    # Label Predictor/Classifier    
    self.labelClassif = Classifier(nb_class, 256)
    self.reco = Reconstruction(nUnit2Reco)

  def siameseDistance(self, inputs, training=False):
    first_elements = inputs[0]
    second_elements = inputs[1]

    femb = self.encoder(first_elements, training=training)
    semb = self.encoder(second_elements, training=training)
    femb = tf.keras.backend.l2_normalize(femb,axis=1)
    semb = tf.keras.backend.l2_normalize(semb,axis=1)
    d_W = tf.reduce_sum( tf.square(femb - semb), axis=1)
    return d_W

  @tf.function(experimental_relax_shapes=True)
  def call(self, inputs, training=False):
    enc_out = self.encoder(inputs, training=training)
    return self.labelClassif(enc_out, training=training), self.reco(enc_out), enc_out