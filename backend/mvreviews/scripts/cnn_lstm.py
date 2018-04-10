import tensorflow as tf
import random
from . import utils
import numpy as np
import os

# check point directory for test
class CNN_LSTM(object):
    def __init__(self):
        checkpoint_dir ="mvreviews/scripts/runs/CNN-LSTM/checkpoints"
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
        graph = tf.Graph()
        with graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
            self.sess = tf.Session(config=config)

            with self.sess.as_default():
                # load saved model and variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(self.sess, checkpoint_file)
                # get input placeholder
                self.input_x = graph.get_operation_by_name("input_placeholder/input_x").outputs[0]
                self.dropout_keep_prob = graph.get_operation_by_name("input_placeholder/dropout_keep_prob").outputs[0]
                self.sent_len = graph.get_operation_by_name("input_placeholder/sentence_length").outputs[0]
                # get prediction node
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]

    def predict(self):

        x_test, y_test, vocab_size, sent_len_test = utils.load_data_and_labels(num_class=10,
                                                                               data_path="mvreviews/data/test/test.txt",
                                                                               labels_path="mvreviews/data/labels.txt",
                                                                               train=False, max_sentence_len=50,
                                                                               max_word_len=200)
        prediction = self.sess.run(self.predictions, feed_dict={self.input_x: x_test, self.dropout_keep_prob: 1.0,
                                                                self.sent_len: sent_len_test})
        return prediction[0]+1
