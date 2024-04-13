
import pickle
import tensorflow as tf

def return_dataset(args, path, length=-1):
    raw_data = pickle.load(open(path, 'rb'))

    """## Create tf.data.Dataset object"""
    data_input = raw_data[:length]
    data_target = raw_data[:length]#[:-1]表示从最后一个数开始取
    
    #pad_sequence()函数将序列转化为经过填充(补零)以后得到的一个长度相同（32）新的序列。
    data_input = tf.keras.preprocessing.sequence.pad_sequences(data_input,maxlen=31, padding='post')#maxlen代表句子的最大长度
    dataset = tf.data.Dataset.from_tensor_slices((data_input, data_input))#<TensorSliceDataset shapes: ((31,), (31,)), types: (tf.int32, tf.int32)>
    dataset = dataset.cache()#cache用于缓存
    dataset = dataset.shuffle(args.shuffle_size).batch(args.bs)#shuffle() 方法将序列的所有元素随机排序
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    #dataset的shape=((31,), (31,))，none代表newaxis，增加一个维度且该维度为1
    return dataset

def return_loader(args):
    """## Load data"""
    train_dataset = return_dataset(args, args.train_save_path, -1)
    test_dataset = return_dataset(args, args.test_save_path, -1)
    
    return train_dataset, test_dataset