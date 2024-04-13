import tensorflow as tf
from models.modules import create_masks, loss_function, create_look_ahead_mask, create_padding_mask
from models.transceiver import sample_batch, mutual_information
from utlis.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity


@tf.function
def gan_train_step(inp, tar, p, net, optim_net, lenmda, channel='AWGN', n_std=0.1, training=False,traingan=False):
    tar_inp = tar[:, :-1]  # exclude the last one,作为目标序列
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    p = tf.random.normal([64,31,16], mean=0.0, stddev=n_std,dtype=tf.float32)
    p = p/tf.norm(p)
    with tf.GradientTape( persistent=True ) as Tape:
        predictions_p,predictions_r,_ = net(inp, tar_inp, p, 40, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=traingan)
        loss = loss_function(tar_real, predictions_r)
        
        g_loss = 10-loss_function(tar_real, predictions_p)
        d_loss = lenmda*loss_function(tar_real, predictions_r) + (1-lenmda)*loss_function(tar_real, predictions_p)
    """
    #先训练语义和信道编解码(冻结g)
    for layer in net.layers:
        if(layer.name=='g'):#看训练哪种模型，记得改对应的名字
            layer.trainable=False
    gradients = Tape.gradient(loss, net.trainable_variables)
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))
    #解冻
    for layer in net.layers:
        if(layer.name=='g'):
            layer.trainable=True
    
    #训练生成器
    g_gradients = Tape.gradient(g_loss, net.trainable_variables[104:108])#同样要看着改
    optim_net.apply_gradients(zip(g_gradients, net.trainable_variables[104:108]))
    """
    #训练鉴别器
    for layer in net.layers:
        if(layer.name=='g' or layer.name=='encoder' or layer.name=='channel_encoder'):#看训练哪种模型，记得改对应的名字
            layer.trainable=False
    d_gradients = Tape.gradient(d_loss, net.trainable_variables)
    optim_net.apply_gradients(zip(d_gradients, net.trainable_variables))
    #解冻
    for layer in net.layers:
        if(layer.name=='g' or layer.name=='encoder' or layer.name=='channel_encoder'):
            layer.trainable=True
    
    return loss,g_loss,d_loss



#用的网络是Transiver_GAN
def eval_step(inp, tar, net, channel='AWGN', n_std=0.1,epsilon=1):
    # loss, loss_mine, mi_numerical = train_step(inp, tar)
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    p = tf.zeros([64,31,16],dtype=tf.float32)
    with tf.GradientTape() as Tape:
        predictions_p, predictions_r,y_r = net(inp, tar_inp, p, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        loss = loss_function(tar_real, predictions_r)#输入扰动是0，经过鉴别器后的结果算出交叉熵
        
    #计算loss关于y的梯度(一个列表，第一项是IndexedSlices，后面都是tensor张量)
    gradients = Tape.gradient(loss, y_r)
    
    #得出对抗样本
    r_list = []
    
    for grad in gradients:
        grad_norm = tf.norm(grad)
        r = tf.cast(epsilon ,dtype = tf.float32)*grad/grad_norm
        r_list.append( r )
        
    r_list = tf.cast(r_list ,dtype = tf.float32)    
    with tf.GradientTape() as Tape:
        predictions_p_1, predictions_r_1,y_r = net(inp, tar_inp, r_list, channel=channel, n_std=n_std,
                                   training=True, enc_padding_mask=enc_padding_mask,
                                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask,traingan=False)
        loss_p = loss_function(tar_real, predictions_p_1)#输入扰动是r_list，经过鉴别器后的结果算出交叉熵
        
    
    return loss,loss_p,predictions_r,predictions_p_1