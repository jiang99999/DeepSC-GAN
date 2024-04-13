import tensorflow as tf
import numpy
from models.modules import create_masks, loss_function, create_look_ahead_mask, create_padding_mask
from models.transceiver import sample_batch, mutual_information
from utlis.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from models.encrypted import key_produce
import math

@tf.function
#加FGM扰动的评估(对Transceiver)
def greedy_decode(args, inp, net,PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx*tf.ones([bs,1], dtype=tf.int32)
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder.call(sema_enc_output)
    ##############################################
    #计算FGM扰动（如果不加FGM就直接用p输入，加FGM用pertutation）
    p = tf.zeros([64,31,16],dtype=tf.float32)
    tar_inp = inp[:, :-1]  # exclude the last one
    tar_real = inp[:, 1:]  # exclude the first one

    enc_padding_mask_2, combined_mask_2, dec_padding_mask_2 = create_masks(inp, tar_inp)
    # semantic encoder
    with tf.GradientTape() as Tape:
        predictions, _, _, y = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask_2,
                                   combined_mask=combined_mask_2, dec_padding_mask=dec_padding_mask_2)
        loss = loss_function(tar_real, predictions)
        
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    with tf.device('CPU:0'):
        gradients = Tape.gradient(loss, y)
    
    #得出对抗样本
    r_list = []
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
          
    r_list = tf.cast(r_list ,dtype = tf.float32)   
    power = tf.norm(r_list.numpy())
    pertutation = r_list/power
    
    ##############################################
    # over the AWGN channel
    if channel == 'AWGN':
        n_std = tf.cast(n_std,dtype=tf.float32)
        PNR = 10**(PNR_dB/10)
        received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)+ n_std*math.sqrt(PNR)*pertutation #PNR
        #用随机噪声替代扰动
        #x = tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)
        #received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32) + n_std*math.sqrt(PNR) * x/tf.norm(x)
        
    elif channel == 'Rician':
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)
    else:
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std)

    for i in range(args.max_length):
        # create sequence padding
        look_ahead_mask = create_look_ahead_mask(tf.shape(outputs)[1])
        dec_target_padding_mask = create_padding_mask(outputs)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # channel decoder
        received_channel_dec_output = net.channel_decoder.call(received_channel_enc_output)
        # semantic deocder
        predictions, _ = net.semantic_decoder.call(outputs, received_channel_dec_output,
                                                   False, combined_mask, enc_padding_mask)
        
        # choose the word from axis = 1
        predictions = predictions[:, -1:, :]  # (batch_size, -1, vocab_size)取第二维的倒数第一个，代表最后一个生成的单词
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        outputs = tf.concat([outputs, predicted_id], axis=-1)#每段句子加上<start>

    return outputs,n_std*math.sqrt(PNR)*pertutation,tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32),channel_enc_output

#不加FGM扰动的评估
def greedy_decode_noattack(args, inp, net,PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx*tf.ones([bs,1], dtype=tf.int32)
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder.call(sema_enc_output)
    p = tf.zeros([64,31,16],dtype=tf.float32)
    # over the AWGN channel
    if channel == 'AWGN':
        n_std = tf.cast(n_std,dtype=tf.float32)
        PNR = 10**(PNR_dB/10)
        received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)+ n_std*math.sqrt(PNR)*p #PNR
    elif channel == 'Rician':
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)
    else:
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std)

    for i in range(args.max_length):
        # create sequence padding
        look_ahead_mask = create_look_ahead_mask(tf.shape(outputs)[1])
        dec_target_padding_mask = create_padding_mask(outputs)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # channel decoder
        received_channel_dec_output = net.channel_decoder.call(received_channel_enc_output)
        # semantic deocder
        predictions, _ = net.semantic_decoder.call(outputs, received_channel_dec_output,
                                                   False, combined_mask, enc_padding_mask)
        
        # choose the word from axis = 1
        predictions = predictions[:, -1:, :]  # (batch_size, -1, vocab_size)取第二维的倒数第一个，代表最后一个生成的单词
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        outputs = tf.concat([outputs, predicted_id], axis=-1)#每段句子加上<start>

    return outputs

#加FGM扰动的评估(对Transceiver_GAN)
def greedy_decode_gan(args, inp, net,PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx*tf.ones([bs,1], dtype=tf.int32)
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder.call(sema_enc_output)
    ##############################################
    #计算FGM扰动（如果不加FGM就直接用p输入，加FGM用pertutation）
    p = tf.zeros([64,31,16],dtype=tf.float32)
    tar_inp = inp[:, :-1]  # exclude the last one
    tar_real = inp[:, 1:]  # exclude the first one

    enc_padding_mask_2, combined_mask_2, dec_padding_mask_2 = create_masks(inp, tar_inp)
    # semantic encoder
    with tf.GradientTape() as Tape:
        predictions_p, predictions_r, y_r = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask_2,
                                   combined_mask=combined_mask_2, dec_padding_mask=dec_padding_mask_2)
        loss = loss_function(tar_real, predictions_r)
        
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    with tf.device('CPU:0'):
        gradients = Tape.gradient(loss, y_r)
    
    #得出对抗样本
    r_list = []
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
          
    r_list = tf.cast(r_list ,dtype = tf.float32)   
    power = tf.norm(r_list.numpy())
    pertutation = r_list/power
    
    ##############################################
    # over the AWGN channel
    if channel == 'AWGN':
        n_std = tf.cast(n_std,dtype=tf.float32)
        PNR = 10**(PNR_dB/10)
        received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)+ n_std*math.sqrt(PNR)*pertutation #PNR
        #用随机噪声替代扰动
        #x = tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)
        #received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32) + n_std*math.sqrt(PNR) * x/tf.norm(x)
        
    elif channel == 'Rician':
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)
    else:
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std)

    for i in range(args.max_length):
        # create sequence padding
        look_ahead_mask = create_look_ahead_mask(tf.shape(outputs)[1])
        dec_target_padding_mask = create_padding_mask(outputs)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # channel decoder
        received_channel_dec_output = net.channel_decoder.call(received_channel_enc_output)
        # semantic deocder
        predictions, _ = net.semantic_decoder.call(outputs, received_channel_dec_output,
                                                   False, combined_mask, enc_padding_mask)
        
        # choose the word from axis = 1
        predictions = predictions[:, -1:, :]  # (batch_size, -1, vocab_size)取第二维的倒数第一个，代表最后一个生成的单词
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        outputs = tf.concat([outputs, predicted_id], axis=-1)#每段句子加上<start>
        
    noa = tf.cast(tf.argmax(predictions_r, axis=-1), tf.int32)

    return outputs,noa,n_std*math.sqrt(PNR)*pertutation,tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32),channel_enc_output

#用的是Transeiver网络
def eval_step_normal(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p,PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)# training代表语义解码加不加dropout
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar_real, predictions)
    
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    #如果信号经过的是瑞利信道，梯度是算不出来的，因为tensorflow不支持复数梯度的计算，而且计算复数的梯度没有实际意义
    #所以只能取高斯分布得到的FGM扰动，再经过一次神经网络
    if channel != 'AWGN':
        with tf.GradientTape() as Tape:
            outs = net(inp, tar_inp, p,PNR_dB, channel='AWGN', n_std=n_std,
                                       training=False, enc_padding_mask=enc_padding_mask,
                                       combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)# training代表语义解码加不加dropout
            p, c, r, y = outs
            loss_temp = loss_function(tar_real, p)
        gradients = Tape.gradient(loss_temp, c)
    else:
        gradients = Tape.gradient(loss, channel_enc_output)

    #得出对抗样本
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)     
    power = tf.norm(r_list.numpy())
    pertutation = r_list/power
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, pertutation, PNR_dB,channel=channel, n_std=n_std,
                   training=False, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions2, channel_enc_output2, received_channel_enc_output2, y2 = outs
        loss_m = loss_function(tar_real, predictions2)
     
    #return loss,loss_m,predictions,predictions2
    return loss,loss_m,predictions,predictions2

#用的是Transeiver网络
def eval_step_normal_pgd(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    pertutation = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, pertutation,PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)# training代表语义解码加不加dropout
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar_real, predictions)
    
    loss_ori = loss
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    gradients = Tape.gradient(loss, y)

    #得出对抗样本
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)     
    power = tf.norm(r_list.numpy())
    
    max = 1
    min = 0
    eps = (max+min)/2
    
    att = []
    for i in range(10):
        #print("第{:d}轮,MAX={:f},MIN={:f}".format(i,max,min))
        with tf.GradientTape() as Tape:
            p = pertutation + eps * r_list/power
        ##############################################
            sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)#（64，31，128）
            # channel encoder
            channel_enc_output = net.channel_encoder.call(sema_enc_output)#（64，31，16）
            if channel == 'AWGN':
                n_std = tf.cast(n_std,dtype=tf.float32)
                PNR = 10**(PNR_dB/10)
                b,w,h = tf.shape(channel_enc_output)
                size = tf.cast(b*w*h ,dtype = tf.float32)
                p = math.sqrt(size)*p
                received_channel_enc_output = channel_enc_output + tf.random.normal(tf.shape(channel_enc_output), mean=0.0, stddev=n_std,dtype=tf.float32)+ n_std*math.sqrt(PNR)*p 
                
            elif channel == 'Rician':
                received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 1, n_std)
            else:
                received_channel_enc_output = net.channel_layer.fading(channel_enc_output, p, PNR_dB, 0, n_std)

            received_channel_dec_output = net.channel_decoder.call(received_channel_enc_output)#（64，31，128）
            # semantic deocder
            predictions2 = net.semantic_decoder.call(tar_inp, received_channel_dec_output,False, combined_mask, dec_padding_mask)
            loss_m = loss_function(tar_real, predictions2)
            
        #print('第{:d}轮相差{:4f}:'.format(i,loss_m-loss))
        if loss_m-loss < 0:
            #print('攻击失败，epsilon是',eps)
            min = eps
            eps = (max+min)/2
            p = pertutation + eps*r_list/power
            
            #print('损失是',loss_m)
        else:
            #print('攻击成功，epsilon是',eps)
            #print('损失是',loss)
            att.append([eps,loss])
            
            max = eps
            
            eps = (max+min)/2
            p = pertutation + eps*r_list/power
            
    #结束循环后取最小的epslion

    if len(att) == 0:
        epsilon = 1
    else:
        epsilon = att[-1][0]
        loss_m = att[-1][1]
    print('epsilon=',epsilon)
    pertutation = p
    
    """
    #计算loss关于y的梯度
    gradients = Tape.gradient(loss_m, received_channel_enc_output)

    #得出对抗样本
    r_list = []

    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )

    r_list = tf.cast(r_list ,dtype = tf.float32)     
    power = tf.norm(r_list.numpy())
    p = pertutation + r_list/power
    """
    
        
    
    return loss_ori,loss_m,predictions,predictions2


#用的是Transeiver-STAR网络
def eval_step_star(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p,PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)# training代表语义解码加不加dropout
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar, predictions)#不太清楚为啥用tar不用tar_real
        
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    if channel != 'AWGN':
        with tf.GradientTape() as Tape:
            outs = net(inp, tar_inp, p,PNR_dB, channel='AWGN', n_std=n_std,
                                       training=False, enc_padding_mask=enc_padding_mask,
                                       combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)# training代表语义解码加不加dropout
            p, c, r, y = outs
            loss_temp = loss_function(tar, p)
        gradients = Tape.gradient(loss_temp, c)
    else:
        gradients = Tape.gradient(loss, channel_enc_output)
    
    #得出对抗样本
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)     
    power = tf.norm(r_list.numpy())
    pertutation = r_list/power
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, pertutation, PNR_dB,channel=channel, n_std=n_std,
                   training=False, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions2, channel_enc_output2, received_channel_enc_output2, y2 = outs
        loss_m = loss_function(tar, predictions2)
        
    
    return loss,loss_m,predictions,predictions2
#在评估时加FGM扰动(Transeiver_gan)
def eval_step_FGM(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape( persistent=True ) as Tape:
        predictions_p,predictions_r,channel_enc_output,y = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        
        loss = loss_function(tar_real, predictions_r)
        
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    if channel != 'AWGN':
        with tf.GradientTape() as Tape:
            pp,pr,c,y2 = net(inp, tar_inp, p,PNR_dB, channel='AWGN', n_std=n_std,
                                       training=False, enc_padding_mask=enc_padding_mask,
                                       combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)# training代表语义解码加不加dropout
            loss_temp = loss_function(tar_real, pr)
        gradients = Tape.gradient(loss_temp, c)
    else:
        gradients = Tape.gradient(loss, y)
    
    #得出对抗样本
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)   
    power = tf.norm(r_list.numpy())
    pertutation = r_list/power
    with tf.GradientTape() as Tape:
        predictions_p_m,predictions_r_m,channel_enc_output_m,y = net(inp, tar_inp, pertutation, PNR_dB, channel=channel, n_std=n_std,
                                   training=False, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        loss_m = loss_function(tar_real, predictions_p_m)
        
    
    return loss,loss_m,predictions_r,predictions_p_m,pertutation