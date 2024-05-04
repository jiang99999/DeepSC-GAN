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
        

        # Employ the perfect CSI here
        if detector == 'LS':
            #with tf.device('GPU:0'):
            h_complex_conj = tf.complex(h_real, -h_imag)
            x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj)
        elif detector == 'MMSE':
            # MMSE Detector
            #with tf.device('GPU:0'):
            h_complex_conj = tf.complex(h_real, -h_imag)
            a = h_complex * h_complex_conj + (n_std * n_std * 2)
            x_est_complex = y_complex * h_complex_conj / a
        else:
            raise ValueError("detector must in LS and MMSE")

        x_est_real = tf.math.real(y_complex)#（64，248）
        x_est_img = tf.math.imag(y_complex)#（64，248）

        x_est_real = tf.expand_dims(x_est_real, -1)
        x_est_img = tf.expand_dims(x_est_img, -1)

        x_est = tf.concat([x_est_real, x_est_img], axis=-1)#（64，248，2）
        x_est = tf.reshape(x_est, (bs, sent_len, -1))#（64，31，16）
        
        return x_est

class Channel_Encoder(tf.keras.Model):
    def __init__(self, size1=256, size2=16):
        super(Channel_Encoder, self).__init__()

        self.dense0 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense1 = tf.keras.layers.Dense(size2, activation=None)
        self.powernorm = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(tf.reduce_mean(tf.square(x)))))

    def call(self, inputs):
        outputs1 = self.dense0(inputs)
        outputs2 = self.dense1(outputs1)
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs

class Channel_Decoder(tf.keras.Model):
    def __init__(self, size1, size2):
        super(Channel_Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense2 = tf.keras.layers.Dense(size2, activation="relu")
        self.dense3 = tf.keras.layers.Dense(size1, activation=None)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, receives):
        x1 = self.dense1(receives)
        x2 = self.dense2(x1)
        x3 = self.dense3(x2)
        output = self.layernorm1(x1 + x3)
        return output

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
        predictions = self.semantic_decoder.call(tar_inp, received_channel_dec_output, combined_mask, training,  dec_padding_mask)
        
        return predictions, channel_enc_output, received_channel_enc_output, received_channel_enc_output

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
        predictions = self.semantic_decoder.call(tar_inp, received_channel_dec_output,training, combined_mask, dec_padding_mask)
        
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

        self.generator = G()

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


    

