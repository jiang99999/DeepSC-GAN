import numpy as np
import tensorflow as tf
#全连接GAN
class G(tf.keras.Model):
    def __init__(self, size1=256,size2=16):
        super(G, self).__init__()
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc1 = tf.keras.layers.Dense(size2,activation=None)
        self.powernorm = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))

    def call(self, inputs):
        x = self.fc0(inputs)
        x = self.fc1(x)
        x = self.powernorm(x)
        
        return x
    
class D(tf.keras.Model):
    def __init__(self, size1=32,size2=16):
        super(D, self).__init__()
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc1 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc2 = tf.keras.layers.Dense(size2,activation=None)
        
    def call(self, inputs):
        x = self.fc0(inputs)
        x = self.fc1(x)
        x = self.fc2(x)
        
        return x
    
#卷积GAN
class G_CNN(tf.keras.Model):
    def __init__(self,):
        super(G_CNN, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(filters=16, kernel_size=16, strides=1, padding='same')
        self.fc = tf.keras.layers.Dense(16,activation=None)
        self.norm = tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True)#层归一化
        self.powernormal = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))#功率归一化

    def call(self, inputs):
        x = self.cnn1(inputs)
        x = self.cnn2(x)
        x = self.norm(x)
        x = self.fc(x)
        x = self.powernormal(x)
        
        return x
    
class D_CNN(tf.keras.Model):
    def __init__(self, size1=128,size2=16):
        super(D_CNN, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(filters=16, kernel_size=8, strides=1, padding='same')
        self.fc = tf.keras.layers.Dense(size1,activation=None)
        self.norm = tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True)#层归一化
        
    def call(self, inputs):
        #print(inputs.shape)
        x = self.cnn1(inputs)
        #print(x.shape)
        x = self.cnn2(x)
        x = self.norm(x)
        x = self.fc(x)
        x = self.norm(x)
        
        return x