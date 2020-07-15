import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from dssm.graph import Graph
import tensorflow as tf
from utils.load_data import load_char_data

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

#p, h, y = load_char_data('input/test.csv', data_size=None)
p, h, y = load_char_data('input/VALI_prefix_title.csv', data_size=None)

model = Graph()
saver = tf.train.Saver()

with tf.Session()as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, '../output/dssm/dssm_9.ckpt')
    loss_eval, acc_eval, recall_eval, f1_eval = sess.run([model.loss, model.acc, model.recall, model.fmeasure],
                         feed_dict={model.p: p, model.h: h, model.y: y, model.keep_prob: 1})

    print('loss_eval: ', loss_eval, ' acc_eval:', acc_eval, ' recall_eval:', recall_eval, ' f1_eval:', f1_eval)
    
    
