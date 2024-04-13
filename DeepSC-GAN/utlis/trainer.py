import tensorflow as tf
import numpy
from models.modules import create_masks, loss_function, create_look_ahead_mask, create_padding_mask
from models.transceiver import sample_batch, mutual_information
from utlis.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from models.encrypted import key_produce

@tf.function
def train_step(inp, tar, net, mine_net, optim_net, optim_mi, channel='AWGN', n_std=0.1, train_with_mine=False):
    # loss, loss_mine, mi_numerical = train_step(inp, tar)
    tar_inp = tar[:, :-1]  # exclude the last one,作为目标序列
    tar_real = tar[:, 1:]  # exclude the first one

    #遮挡一批序列中所有的填充标记（pad tokens）。这确保了模型不会将填充作为输入。该 mask 表明填充值 0 出现的位置：在这些位置 mask 输出 1，否则输出 0
    #mask在点乘时乘以很小的量，使softmax输出很小，padding_mask是为了让每句话补零的部分不参与反向传输影响结果
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    #shape----enc_padding_mask:(64, 1, 1, 31)：输入数据的填充遮挡
    #combined_mask:(64, 1, 30, 30)：前檐遮挡,只让让之前的序列参与预测
    #dec_padding_mask:(64, 1, 1, 31)：译码时的填充遮挡
    
    with tf.GradientTape() as Tape:
        # semantic encoder
        #inp和tar的shape是(64, 31)和(64, 30)，inp是编码器的输入（完整的嵌入向量），tar是解码输入（去掉最后一位表示第N-1时刻的输入）
        outs = net(inp, tar_inp, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        #predictions尺寸(64, 30, 22234)
        predictions, channel_enc_output, received_channel_enc_output = outs
        loss_error = loss_function(tar_real, predictions)
        loss = loss_error
        #计算输入信道前和输出信道后的互信息量
        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)
        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb#loss_mine代表计算互信息量网络（mutual information network）的损失
        if train_with_mine:
            loss = loss_error + 0.05 * loss_mine
    #计算loss关于可训练变量的梯数
    gradients = Tape.gradient(loss, net.trainable_variables)#trainable_variables代表可训练的变量，当trainable设定为True时便能调用
    # updata gradients（根据net）
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))

    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, channel=channel, n_std=0.1,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)

        predictions, channel_enc_output, received_channel_enc_output = outs

        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)

        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb

        mi_numerical = 2.20

    #计算loss_mine关于可训练变量的梯数
    gradients = Tape.gradient(loss_mine, mine_net.trainable_variables)
    # updata gradients（根据mine_net）
    optim_mi.apply_gradients(zip(gradients, mine_net.trainable_variables))#zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组
    
    return loss, loss_mine, mi_numerical

#不加扰动的训练
def train_step_noattack(inp, tar, p, net, optim_net, channel='AWGN', n_std=0.1, train_with_mine=False, epsilon=1):
    tar_inp = tar[:, :-1]  # exclude the last one,作为目标序列
    tar_real = tar[:, 1:]  # exclude the first one
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
       
    
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, 0 ,channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)#training代表是否使用dropout训练
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar_real, predictions)
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    gradients = Tape.gradient(loss, net.trainable_variables)
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))    
    
    return loss

#加FGM扰动的对抗训练
def train_attack_step(inp, tar, p, PNR_dB, net, optim_net, channel='AWGN', n_std=0.1, train_with_mine=False, epsilon=1):
    tar_inp = tar[:, :-1]  # exclude the last one,作为目标序列
    tar_real = tar
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
       
    
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        
        predictions, channel_enc_output, received_channel_enc_output, y = outs
 
        loss = loss_function(tar_real, predictions)
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    gradients = Tape.gradient(loss, y)
    #得出对抗样本
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

#加权重扰动的对抗训练
def train_attack_step_2(inp, tar, p, net, optim_net, channel='AWGN', n_std=0.1, train_with_mine=False, epsilon=1):
    tar_inp = tar[:, :-1]  # exclude the last one,作为目标序列
    tar_real = tar[:, 1:]  # exclude the first one
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
       
    
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions, channel_enc_output, received_channel_enc_output, received_channel_enc_output = outs
        loss = loss_function(tar_real, predictions)
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    gradients = Tape.gradient(loss, net.trainable_variables)
    """
    #第一项IndexedSlices需要特殊处理
    grad_norm = tf.norm(gradients[0].values)
    r = gradients[0].values/grad_norm
    gradients_first = gradients[0].values + r
    optim_net.apply_gradients(zip(gradients, net.trainable_variables[0]))
    """ 
    #克隆原始模型的权重
    net_ori = net
    #计算对抗样本的权重
    data_list = []
    data_list.append(net.trainable_variables[0])
    i = 1
    for grad in gradients[1:]:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        data = net.trainable_variables[i] + r
        data_list.append(data)
        i = i + 1
    #将权重赋值在原来的模型上    
    i = 0
    for data in data_list:
        #print(data.numpy())
        data_list[i]=data.numpy()
        i=i+1
    #print(net.layers[0].get_weights())
    net.layers[0].set_weights(data_list[0:37])#0:37表示第1个到第37个
    net.layers[1].set_weights(data_list[37:104])
    net.layers[2].set_weights(data_list[104:108])
    net.layers[3].set_weights(data_list[108:116])
    
    
    #计算对抗样本的损失
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions, channel_enc_output, received_channel_enc_output,received_channel_enc_output = outs
        loss_m = loss_function(tar_real, predictions)
    
    
    #算对抗样本的梯度
    gradients = Tape.gradient(loss_m, net.trainable_variables)
    #复原net
    net = net_ori
    """
    j = 0
    for layers in net.layers:
        weights = net_ori[j].get_weights()
        layers.set_weights(weights)
        j = j + 1
    """
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))
    
    return loss, loss_m

#这是正经的评估
def greedy_decode(args, inp, net, channel='AWGN', n_std=0.1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx*tf.ones([bs,1], dtype=tf.int32)
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder.call(sema_enc_output)
    # over the AWGN channel
    if channel == 'AWGN':
        received_channel_enc_output = net.channel_layer.awgn(channel_enc_output, n_std)
    elif channel == 'Rician':
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 1, n_std)
    else:
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 0, n_std)

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

#用的是Transeiver网络
def eval_step_normal(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, p,PNR_dB, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions, channel_enc_output, received_channel_enc_output, y = outs
        loss = loss_function(tar_real, predictions)
        
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
    pertutation = r_list/power
    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, pertutation, PNR_dB,channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)
        predictions2, channel_enc_output2, received_channel_enc_output2, y2 = outs
        loss_m = loss_function(tar_real, predictions2)
        
    
    return loss,loss_m,predictions,predictions2

#在评估时加FGM扰动
def eval_step_FGM(inp, tar, net, PNR_dB, channel='AWGN', n_std=0.1,epsilon=1):
    
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        predictions_p,predictions_r,y = net(inp, tar_inp, p, PNR_dB, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        
        loss = loss_function(tar_real, predictions_r)
        
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
    pertutation = r_list/power
    with tf.GradientTape() as Tape:
        predictions_p_m,predictions_r_m,y = net(inp, tar_inp, pertutation, PNR_dB, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        loss_m = loss_function(tar_real, predictions_r_m)
        
    
    return loss,loss_m,predictions_r,predictions_r_m,pertutation