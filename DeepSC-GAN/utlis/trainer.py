import tensorflow as tf
import numpy
from models.modules import create_masks, loss_function, create_look_ahead_mask, create_padding_mask
from models.transceiver import sample_batch, mutual_information
from utlis.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from models.encrypted import key_produce

@tf.function


#Normal training(no attack)
def train_step_noattack(inp, tar, p, net, optim_net, channel='AWGN', n_std=0.1, train_with_mine=False, epsilon=1):
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
       
    
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, 0 ,channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar_real, predictions)
    gradients = Tape.gradient(loss, net.trainable_variables)
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))    
    
    return loss

#Adversial training
def train_attack_step(inp, tar, p, PNR_dB, net, optim_net, channel='AWGN', n_std=0.1, train_with_mine=False, epsilon=1):
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
       
    
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        
        predictions, channel_enc_output, received_channel_enc_output, y = outs
 
        loss = loss_function(tar_real, predictions)
    gradients = Tape.gradient(loss, y)
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)    
    r_list = r_list/tf.norm(r_list)
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, r_list, PNR_dB, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss_m = loss_function(tar_real, predictions)
        
    gradients = Tape.gradient(loss_m, net.trainable_variables)    
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))    
    
    return loss,loss_m

