import numpy as np
import tensorflow as tf
#加密器（改词向量）
class Encryptor(tf.keras.Model):
    def __init__(self, size1=32,size2=3968):
        super(Encryptor, self).__init__()
        self.depth = 5
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")#rule（x）=max(0，x)
        self.fc = [tf.keras.layers.Dense(size1,activation=None) for _ in range(self.depth)]
        self.fc1 = tf.keras.layers.Dense(size2,activation="relu")
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, key):
        
        cipher = tf.concat([inputs,key],1)
        cipher = tf.reshape(cipher,shape=[64,1,4096])
        x = self.fc0(cipher)
        for idx, layer in enumerate(self.fc):
            if idx == 0:
                x = tf.nn.relu(layer(x))
            else:
                x = tf.nn.relu(x + layer(x))
        x = self.fc1(x)
        x = tf.reshape(x,shape=[64,31,128])
        x = self.layernorm(x)
        return x

#解密器（改词向量）
class Decryptor(tf.keras.Model):
    def __init__(self, size1=32,size2=3968):
        super(Decryptor, self).__init__()
        self.depth = 5
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc = [tf.keras.layers.Dense(size1,activation=None) for _ in range(self.depth)]
        self.fc1 = tf.keras.layers.Dense(size2,activation="relu")
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self, receives, key):
   
        text = tf.concat([receives,key],1)
        text = tf.reshape(text,shape=[64,1,4096])
        x = self.fc0(text)
        for idx, layer in enumerate(self.fc):
            if idx == 0:
                x = tf.nn.relu(layer(x))
            else:
                x = tf.nn.relu(x + layer(x))
        x = self.fc1(x)
        x = tf.reshape(x,shape=[64,31,128])
        x = self.layernorm(x)
        
        return x
"""
#加密器（改词）
class Encryptor_1(tf.keras.Model):
    def __init__(self, size1=32,size2=3968):
        super(Encryptor, self).__init__()
        self.depth = 5
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")#rule（x）=max(0，x)
        

    def call(self, inputs, key):
        #key = tf.random.uniform(shape=result.shape, minval=0, maxval=2, dtype=tf.int32)
        cipher = tf.concat([inputs,key],1)
        cipher = tf.reshape(cipher,shape=[64,1,4096])
        x = self.fc0(cipher)
        for idx, layer in enumerate(self.fc):
            if idx == 0:
                x = tf.nn.relu(layer(x))
            else:
                x = tf.nn.relu(x + layer(x))
        x = self.fc1(x)
        x = tf.reshape(x,shape=[64,31,128])
        x = self.layernorm(x)
        return x

#解密器（改词）
class Decryptor_1(tf.keras.Model):
    def __init__(self, size1=32,size2=3968):
        super(Decryptor, self).__init__()
        self.depth = 5
        self.fc0 = tf.keras.layers.Dense(size1,activation="relu")
        
    def call(self, receives, key):
   
        rows = 5  # 矩阵的行数
        cols = 5  # 矩阵的列数
        # 生成随机的0和1组成的矩阵
        matrix = np.random.randint(2, size=(rows, cols))
        matrix = tf.cast(matrix, dtype=tf.int32)
        
        mask = tf.math.logical_not(tf.math.equal(inp, 0))#pad
        matrix = tf.where(tf.math.equal(inp, 2), x=0, y=matrix, name=None)#结尾置0（令牌2）
        matrix = matrix[:, 1:]#除去第一个
        return x
"""    
    
def key_produce(inp):#model-1：在原始数据上全部叠加1（不处理首尾符号和pading）
    mask = tf.math.logical_not(tf.math.equal(inp, 0))#pad
    #sed = tf.random.uniform(shape=[64,31],minval=0,maxval=22234,dtype=tf.int32,name=None)
    sed = tf.ones(shape=[64,31],dtype=tf.int32, name=None)
    sed=tf.where(tf.math.equal(inp, 2), x=0, y=sed, name=None)#结尾置0（令牌2）
    mask=tf.cast(mask,dtype=tf.int32)
    sed*=mask
    sed = sed[:, 1:]#除去第一个
    zero = tf.zeros([64,1],dtype=tf.int32)
    sed = tf.concat([zero,sed],1)
    key = tf.where(sed+inp>=22234, x=0, y=sed+inp-22229, name=None)#如果产生的密文大于词汇表数，则该位清零
    chiper = key+inp
    return chiper,key

def key_produce_2(inp):#model-2:在原始数据上随机产生0或1的叠加信号（不处理首尾符号和pading）
    #随机产生密钥key
    inp = tf.where(tf.math.equal(inp, 2), x=0, y=inp, name=None)#结尾置0（令牌2）
    mask = tf.math.logical_not(tf.math.equal(inp, 0))#pad
    mask = tf.cast(mask, dtype=tf.int32)
    inp*=mask
    rows = 64  # 矩阵的行数
    cols = 31  # 矩阵的列数
    # 生成随机的0和1组成的矩阵
    matrix = np.random.randint(2, size=(rows, cols))
    matrix *= mask
    matrix = matrix[:, 1:]#除去第一个
    zero = tf.zeros([64,1],dtype=tf.int32)
    key = tf.concat([zero,matrix],1)
    #产生密文chiper
    chiper = key+inp
    return chiper,key

def decrypted(chiper,key):
    key = key[:, 1:]
    text = chiper-key
    return text