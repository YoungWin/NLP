import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dssm.graph import Graph
import tensorflow as tf
from utils.load_data import load_char_data
from dssm import args

#p, h, y = load_char_data('input/train.csv', data_size=None)    # 做成文本字粒度的index编码，整个文本编码完的长度 做了padding
#p_eval, h_eval, y_eval = load_char_data('input/dev.csv', data_size=None)

p, h, y = load_char_data('input/TRAIN_ALL.csv', data_size=None)    # 做成文本字粒度的index编码，整个文本编码完的长度 做了padding
p_eval, h_eval, y_eval = load_char_data('input/VALI_ALL.csv', data_size=None)

p_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
h_holder = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
y_holder = tf.placeholder(dtype=tf.int32, shape=None, name='y')

dataset = tf.data.Dataset.from_tensor_slices((p_holder, h_holder, y_holder))
dataset = dataset.batch(args.batch_size).repeat(args.epochs)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

model = Graph()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9

with tf.Session(config=config)as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(iterator.initializer, feed_dict={p_holder: p, h_holder: h, y_holder: y})   # iterator.initializer 先对迭代器初始化
    steps = int(len(y) / args.batch_size)   # 每个epoch多少个batch
    for epoch in range(args.epochs):
        for step in range(steps):
            p_batch, h_batch, y_batch = sess.run(next_element)
            _, loss, acc, recall, f1 = sess.run([model.train_op, model.loss, model.acc, model.recall, model.fmeasure],
                                    feed_dict={model.p: p_batch, model.h: h_batch, model.y: y_batch, model.keep_prob: args.keep_prob})
            if step % 50 == 0:
                print('epoch:', epoch, ' step:', step, ' loss: ', loss, ' acc:', acc, ' recall:', recall, ' f1:', f1)

        loss_eval, acc_eval, recall_eval, f1_eval = sess.run([model.loss, model.acc, model.recall, model.fmeasure],
                                       feed_dict={model.p: p_eval, model.h: h_eval, model.y: y_eval, model.keep_prob: 1})
        print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval, ' recall_eval:', recall_eval, ' f1_eval:', f1_eval)
        print('\n')
        saver.save(sess, f'../output/dssm/dssm_{epoch}.ckpt')