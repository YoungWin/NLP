import tensorflow as tf
from dssm import args


class Graph:
    def __init__(self):
        self.p = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='p')
        self.h = tf.placeholder(dtype=tf.int32, shape=(None, args.seq_length), name='h')
        self.y = tf.placeholder(dtype=tf.int32, shape=None, name='y')
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='drop_rate')
        self.embedding = tf.get_variable(dtype=tf.float32, shape=(args.vocab_size, args.char_embedding_size), name='embedding')   # 完整的嵌入张量
        self.forward()

    def dropout(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob)

    def fully_connect(self, x):
        x = tf.layers.dense(x, 128, activation='tanh')   # 全连接层
        x = self.dropout(x)
        x = tf.layers.dense(x, 256, activation='tanh')
        x = self.dropout(x)
        x = tf.layers.dense(x, 128, activation='tanh')
        x = self.dropout(x)
        x = tf.reshape(x, shape=(-1, x.shape[1] * x.shape[2]))   #
        return x

    @staticmethod
    def cosine(p, h):
        p_norm = tf.norm(p, axis=1, keepdims=True)
        h_norm = tf.norm(h, axis=1, keepdims=True)
        cosine = tf.reduce_sum(tf.multiply(p, h), axis=1, keepdims=True) / (p_norm * h_norm)
        return cosine

    def forward(self):
        p_embedding = tf.nn.embedding_lookup(self.embedding, self.p)
        h_embedding = tf.nn.embedding_lookup(self.embedding, self.h)
        p_context = self.fully_connect(p_embedding)
        h_context = self.fully_connect(h_embedding)
        # [0,1],[1,0]  [0,0,1]...
        pos_result = self.cosine(p_context, h_context)
        neg_result = 1 - pos_result
        logits = tf.concat([neg_result, pos_result], axis=1)
        self.train(logits)

    def train(self, logits):
        y = tf.one_hot(self.y, args.class_size)   # onehot矩阵维度：y的第一个维度*2列
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits)
        self.loss = tf.reduce_mean(loss)
        self.train_op = tf.train.AdamOptimizer(args.learning_rate).minimize(self.loss)
        #prediction = tf.argmax(logits, axis=1)   # neg类 预测0
        #correct_prediction = tf.equal(tf.cast(prediction, tf.int32), self.y)
        #self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        predicted = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        actual = tf.cast(self.y, tf.int32)

        # Count true positives, true negatives, false positives and false negatives.
        tp = tf.count_nonzero(tf.multiply(predicted, actual))
        tn = tf.count_nonzero(tf.multiply(predicted - 1, actual - 1))
        fp = tf.count_nonzero(tf.multiply(predicted, actual - 1))
        fn = tf.count_nonzero(tf.multiply(predicted - 1, actual))

        # Calculate accuracy, precision, recall and F1 score.
        self.acc = (tp + tn) / (tp + fp + fn + tn)
        self.precision = tp / (tp + fp)
        self.recall = tp / (tp + fn)
        self.fmeasure = (2 * self.precision * self.recall) / (self.precision + self.recall)

