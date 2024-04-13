from models.modules import Encoder, Decoder ,SEncoder, SDecoder,SE,SD
from models.gan import G,D,G_CNN,D_CNN
import tensorflow as tf
import math

def complexmulty(a,b):
    a_real = tf.math.real(a)
    a_imag = tf.math.imag(a)
    b_real = tf.math.real(b)
    b_imag = tf.math.imag(b)
    return a
    
class Channels(tf.keras.Model):
    def __init__(self):
        super(Channels, self).__init__()
        
    def call(self, inputs, p, PNR_dB,n_std=0.1,channel='AWGN',K=0,detector='MMSE'):
        if channel=='AWGN':
            return self.awgn(inputs, p, PNR_dB,n_std)
        elif channel=='Rayleigh':
            return self.fading(inputs, p, PNR_dB, 0, n_std, detector)
        else:
            return self.fading(inputs, p, PNR_dB, 1, n_std, detector)
            
    def awgn(self, inputs, p, PNR_dB,n_std=0.1):
        x = inputs
        #把n_std的dtype改成tf.float32,
        n_std = tf.cast(n_std,dtype=tf.float32)
        PNR = 10**(PNR_dB/10)
        b,w,h = tf.shape(x)
        size = tf.cast(b*w*h ,dtype = tf.float32)
        p = math.sqrt(size)*p
        y = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=n_std,dtype=tf.float32) + n_std*math.sqrt(PNR)*p #
        return y

    def fading(self, inputs, p, PNR_dB, K=0, n_std=0.1, detector='MMSE'):

        x = inputs
        bs, sent_len, d_model = x.shape
        mean = math.sqrt(K / (2 * (K + 1)))
        std = math.sqrt(1 / (2 * (K + 1)))
        #将信号分成实部和虚部传输
        x = tf.reshape(x, (bs, -1, 2))
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        
        
        x_complex = tf.complex(x_real, x_imag)
        # create the fading factor
        h_real = tf.random.normal((1,), mean=mean, stddev=std)
        h_imag = tf.random.normal((1,), mean=mean, stddev=std)
        h_complex = tf.complex(h_real, h_imag)
        # create the noise vector
        n = tf.random.normal(tf.shape(x), mean=0.0, stddev=n_std)
        n_real = n[:, :, 0]
        n_imag = n[:, :, 1]
        n_complex = tf.complex(n_real, n_imag)
        # Transmit Signals
        y_complex = tf.multiply(x_complex, h_complex) + n_complex#(64,248)
        
        """
        # Employ the perfect CSI here
        if detector == 'LS':
            #用CPU计算共轭，jupyter使用tf.math.conj会报GPU错误
            #with tf.device('GPU:0'):
            h_complex_conj = tf.complex(h_real, -h_imag)
            x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj)
        elif detector == 'MMSE':
            # MMSE Detector
            #用CPU计算共轭，jupyter使用tf.math.conj会报GPU错误
            #with tf.device('GPU:0'):
            h_complex_conj = tf.complex(h_real, -h_imag)
            a = h_complex * h_complex_conj + (n_std * n_std * 2)
            x_est_complex = y_complex * h_complex_conj / a
        else:
            raise ValueError("detector must in LS and MMSE")
        """
        x_est_real = tf.math.real(y_complex)#（64，248）
        x_est_img = tf.math.imag(y_complex)#（64，248）

        x_est_real = tf.expand_dims(x_est_real, -1)
        x_est_img = tf.expand_dims(x_est_img, -1)

        x_est = tf.concat([x_est_real, x_est_img], axis=-1)#（64，248，2）
        x_est = tf.reshape(x_est, (bs, sent_len, -1))#（64，31，16）

        # method 1
        noise_level = n_std * tf.ones((bs, sent_len, 1))
        h_real = h_real * tf.ones((bs, sent_len, 1))
        h_imag = h_imag * tf.ones((bs, sent_len, 1))
        h = tf.concat((h_real, h_imag), axis=-1)
        out1 = tf.concat((h, x_est), -1)   # [bs, sent_len, 2 + d_model]

        # method 2
        y_complex_real = tf.math.real(y_complex)
        y_complex_img = tf.math.imag(y_complex)
        y_complex_real = tf.expand_dims(y_complex_real, -1)
        y_complex_img = tf.expand_dims(y_complex_img, -1)
        y = tf.concat([y_complex_real, y_complex_img], axis=-1)
        y = tf.reshape(y, (bs, sent_len, -1))
        out2 = tf.concat((h, y), -1)  # [bs, sent_len, 2 + d_model]
        
        return x_est
    


#信道编码（256+16）
class Channel_Encoder(tf.keras.Model):
    def __init__(self, size1=256, size2=16):
        super(Channel_Encoder, self).__init__()

        self.dense0 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense1 = tf.keras.layers.Dense(size2, activation=None)
        self.powernorm = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))))

    def call(self, inputs):
        outputs1 = self.dense0(inputs)
        outputs2 = self.dense1(outputs1)
        # POWER = tf.sqrt(tf.reduce_mean(tf.square(outputs2)))把16长度的向量归一化成单位向量
        #tf.reduce_mean用于计算所有元素的均值
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs

#信道译码（128+512+128）
class Channel_Decoder(tf.keras.Model):
    def __init__(self, size1, size2):
        super(Channel_Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense2 = tf.keras.layers.Dense(size2, activation="relu")
        # size2 equals to d_model
        self.dense3 = tf.keras.layers.Dense(size1, activation=None)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)#层归一化

    def call(self, receives):
        x1 = self.dense1(receives)
        x2 = self.dense2(x1)
        x3 = self.dense3(x2)

        output = self.layernorm1(x1 + x3)#残差连接
        return output

#用卷积设置信道编码
class Channel_Encoder_CNN(tf.keras.Model):
    def __init__(self, size1=256, size2=16):
        super(Channel_Encoder_CNN, self).__init__()
        
        self.cnn1 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(filters=128, kernel_size=2, strides=1, padding='same')
        self.fc1 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc2 = tf.keras.layers.Dense(size2,activation="relu")
        self.fc3 = tf.keras.layers.Dense(size2,activation=None)
        self.powernormal = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))#功率归一化

    def call(self, inputs):
        outputs1 = self.cnn1(inputs)
        outputs1 = tf.nn.relu(outputs1)
        
        outputs2 = self.cnn2(outputs1)
        outputs2 = tf.nn.relu(outputs2 + outputs1)
        
        outputs3 = self.fc1(outputs2)
        outputs4 = self.fc2(outputs3)
        outputs5 = self.fc3(outputs4)
        power_norm_outputs = self.powernormal(outputs5)
        return power_norm_outputs

#信道译码（128+512+128）
class Channel_Decoder_CNN(tf.keras.Model):
    def __init__(self, size1, size2):
        super(Channel_Decoder_CNN, self).__init__()
        self.cnn1 = tf.keras.layers.Conv1D(filters=size2, kernel_size=2, strides=1, padding='same')
        self.cnn2 = tf.keras.layers.Conv1D(filters=size2, kernel_size=2, strides=1, padding='same')
        self.fc1 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc2 = tf.keras.layers.Dense(size1,activation="relu")
        self.fc3 = tf.keras.layers.Dense(size1,activation=None)
        self.norm = tf.keras.layers.LayerNormalization(axis=1 , center=True , scale=True)#层归一化
        
    def call(self, receives):
        x1 = self.cnn1(receives)
        x1 = tf.nn.relu(x1)
        
        x2 = self.cnn2(x1)
        x2 = tf.nn.relu(x2 + x1)
        
        x3 = self.fc1(x2)
        x4 = self.fc2(x3)
        x5 = self.fc3(x4)
        
        output = self.norm(x5)
        
        return output
#设计模型
class Mine(tf.keras.Model):
    def __init__(self, hidden_size=10):
        super(Mine, self).__init__()
        randN_05 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        bias_init = tf.keras.initializers.Constant(0)

        self.dense1 = tf.keras.layers.Dense(hidden_size, bias_initializer=bias_init, kernel_initializer=randN_05,
                                            activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden_size, bias_initializer=bias_init, kernel_initializer=randN_05,
                                            activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)

    def call(self, inputs):
        output1 = self.dense1(inputs)
        output2 = self.dense2(output1)
        output = self.dense3(output2)
        #        output1 = self.dense1(inputs)
        #        output2 = self.dense2(output1)
        #        output = self.dense3(output2)
        return output

#互信息量
def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = tf.exp(mine_net(marginal))
    mi_lb = tf.reduce_mean(t) - tf.math.log(tf.reduce_mean(et))#reduce_mean()用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = tf.cast(joint, dtype=tf.float32)
    marginal = tf.cast(marginal, dtype=tf.float32)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * tf.reduce_mean(et)

    # unbiasing use moving average
    loss = -(tf.reduce_mean(t) - (1 / tf.reduce_mean(ma_et)) * tf.reduce_mean(et))
    # use biased estimator
    # loss = - mi_lb
    return loss, ma_et, mi_lb

#
def sample_batch(rec, noise):
    rec = tf.reshape(rec, shape=[-1, 1])
    # noise = noise[:, :, 2:]
    noise = tf.reshape(noise, shape=[-1, 1])
    rec_sample1, rec_sample2 = tf.split(rec, 2, 0)
    noise_sample1, noise_sample2 = tf.split(noise, 2, 0)
    joint = tf.concat([rec_sample1, noise_sample1], 1)
    marg = tf.concat([rec_sample1, noise_sample2], 1)
    return joint, marg

#不带生成器和鉴别器的模型
class Transeiver(tf.keras.Model):
    def __init__(self, args):
        super(Transeiver, self).__init__()
        
        # semantic encoder
        self.semantic_encoder = Encoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)
        # semantic decoder
        self.semantic_decoder = Decoder(args.decoder_num_layer, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)
        # channel encoder
        self.channel_encoder = Channel_Encoder(256, 16)
        
        # channel decoder
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channels()
        

    def call(self, inputs, tar_inp, p, PNR_dB, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        
        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)#（64，31，128）
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)#（64，31，16）
        # over the AWGN channel
        """
        if channel=='AWGN':
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, p, PNR_dB, n_std)#（64，31，16）
        elif channel=='Rayleigh':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std, 'LS')
            
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)
        """
        received_channel_enc_output = self.channel_layer(channel_enc_output, p, PNR_dB, n_std,channel)
        # channel decoder
        received_channel_dec_output = self.channel_decoder.call(received_channel_enc_output)#（64，31，128）
        
        # semantic deocder
        predictions = self.semantic_decoder.call(tar_inp, received_channel_dec_output,
                                                    training, combined_mask, dec_padding_mask)#（64，31，22234）
        #语义解码器与Transformer解码相似，同时输入发射端的文本和信道解码的输出
        
        return predictions, channel_enc_output, received_channel_enc_output, received_channel_enc_output
    
class Transeiver_star(tf.keras.Model):
    def __init__(self, args):
        super(Transeiver_star, self).__init__()
        
        # semantic encoder
        self.semantic_encoder = SEncoder(args.cycle_num ,args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = SDecoder(args.cycle_num ,args.decoder_num_layer, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)
        # channel encoder
        self.channel_encoder = Channel_Encoder(256, 16)
        
        # channel decoder
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channels()
        

    def call(self, inputs, tar_inp, p, PNR_dB, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        
        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)#（64，31，128）
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)#（64，31，16）
        # over the AWGN channel
        if channel=='AWGN':
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, p, PNR_dB, n_std)#（64，31，16）
        elif channel=='Rayleigh':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std)
            
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)

        # channel decoder
        received_channel_dec_output = self.channel_decoder.call(received_channel_enc_output)#（64，31，128）
        # semantic deocder
        predictions = self.semantic_decoder.call(tar_inp, received_channel_dec_output, combined_mask, training,  dec_padding_mask)#（64，31，22234）
        #语义解码器与Transformer解码相似，同时输入发射端的文本和信道解码的输出
        
        return predictions, channel_enc_output, received_channel_enc_output, received_channel_enc_output
    
#不加FFN的star-transformer
class Transeiver_Star(tf.keras.Model):
    def __init__(self, args):
        super(Transeiver_Star, self).__init__()
        
        # semantic encoder
        self.semantic_encoder = SE(args.cycle_num ,args.cycle_layers, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = SD(args.cycle_num ,args.cycle_layers, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)
        # channel encoder
        self.channel_encoder = Channel_Encoder(256, 16)
        
        # channel decoder
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channels()
        

    def call(self, inputs, tar_inp, p, PNR_dB, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        
        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)#（64，31，128）
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)#（64，31，16）
        # over the AWGN channel
        
        received_channel_enc_output = self.channel_layer(channel_enc_output, p, PNR_dB, n_std,channel)
        # channel decoder
        received_channel_dec_output = self.channel_decoder.call(received_channel_enc_output)#（64，31，128）
        # semantic deocder
        predictions = self.semantic_decoder.call(tar_inp, received_channel_dec_output,training, combined_mask, dec_padding_mask)#（64，31，22234）
        #语义解码器与Transformer解码相似，同时输入发射端的文本和信道解码的输出
        
        return predictions, channel_enc_output, received_channel_enc_output, received_channel_enc_output
    
class Transeiver_GAN(tf.keras.Model):
    def __init__(self, args):
        super(Transeiver_GAN, self).__init__()
        
        # semantic encoder
        self.semantic_encoder = Encoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = Decoder(args.decoder_num_layer, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)
        
        #生成器
        self.generator = G()
        #鉴别器()
        #self.discriminator = D_CNN()
        
        # channel encoder
        self.channel_encoder = Channel_Encoder(256, 16)
        
        # channel decoder
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channels()
        

    def call(self, inputs, tar_inp, pertutation,PNR_dB, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None,traingan=False):
        
        
        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)#（64，31，128）
        
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)#（64，31，16）
        
        #生成器生成扰动
        if(traingan):
            p = self.generator.call(channel_enc_output)
        else:
            p = pertutation
        # over the AWGN channel
        y_p = self.channel_layer(channel_enc_output, p, PNR_dB, n_std,channel)
        y_r = self.channel_layer(channel_enc_output, tf.zeros([64,31,16]), PNR_dB, n_std,channel)
        # channel decoder
        received_channel_dec_output_p = self.channel_decoder.call(y_p)#（64，31，128）
        received_channel_dec_output_r = self.channel_decoder.call(y_r)
        # semantic deocder
        predictions_p = self.semantic_decoder.call(tar_inp, received_channel_dec_output_p,
                                                    training, combined_mask, dec_padding_mask)#（64，31，22234）
        predictions_r = self.semantic_decoder.call(tar_inp, received_channel_dec_output_r,
                                                    training, combined_mask, dec_padding_mask)
        
        
        
        return predictions_p,predictions_r,channel_enc_output,y_r



#微调Transeiver最后一层输出（22234-->39023）

    

