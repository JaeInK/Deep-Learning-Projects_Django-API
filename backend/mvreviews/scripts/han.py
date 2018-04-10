import tensorflow as tf
import numpy as np
import random
from . import utils


# check point directory for test
class HAN():
    def __init__(self):
        checkpoint_dir = "mvreviews/scripts/runs/HAN/checkpoints"
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
                self.word_len = graph.get_operation_by_name("input_placeholder/word_lengths").outputs[0]
                # get prediction node
                self.predictions = graph.get_operation_by_name("output/predictions").outputs[0]


    def predict(self):
        x_test, y_test, vocab_size, sent_len_test = utils.load_data_and_labels(num_class=10,
                                                                               data_path="mvreviews/data/test/test.txt",
                                                                               labels_path="mvreviews/data/test/labels.txt",
                                                                               train=False, max_sentence_len=50,
                                                                               max_word_len=100)

        x = x_test
        X = np.reshape(x, [x.shape[0] * x.shape[1], x.shape[2]])
        zeros = np.zeros([X.shape[0], 1])
        word_len_test = np.sum(X != zeros, axis=1)
        word_len_test = np.reshape(word_len_test, [x.shape[0], x.shape[1]])
        prediction = self.sess.run(self.predictions, feed_dict={self.input_x: x_test, self.word_len: word_len_test,
                                                                self.sent_len: sent_len_test,
                                                                self.dropout_keep_prob: 1.0})

        return prediction[0] + 1

