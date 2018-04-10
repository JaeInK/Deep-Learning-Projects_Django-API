import tensorflow as tf
from six.moves import cPickle
from six import text_type
from .model import Model

import os

#input - start text
def main(input):
    save_dir = 'save'
    n = 400
    prime = input
    sample = 1
    
    return generate(save_dir, n, prime, sample)

def generate(save_dir, n, prime, sample):

    with open(os.path.join("generation", save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)
        
        # config.pkl -->
            # {'batch_size': 64,
            #  'data_dir': 'data',
            #  'grad_clip': 5.0,
            #  'init_from': None,
            #  'input_keep_prob': 0.8,
            #  'log_dir': 'logs',
            #  'model': 'lstm',
            #  'num_epochs': 20,
            #  'num_layers': 4,
            #  'output_keep_prob': 0.8,
            #  'rnn_size': 300,
            #  'save_dir': 'save',
            #  'save_every': 1000,
            #  'seq_length': 15,
            #  'vocab_size': 31869}
    with open(os.path.join("generation", save_dir, 'chars_vocab.pkl'), 'rb') as f:
        chars, vocab = cPickle.load(f)
    model = Model(saved_args, training=False)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(os.path.join("generation",save_dir))
        #ckpt = tf.train.latest_checkpoint(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #ret, score = model.sample(sess, chars, vocab, args.n, args.prime, args.sample)
            try:
                sentence = model.sample(sess, chars, vocab, n, prime, sample)
            except:
                sentence = "하나의 형태소 단위를 입력해 주세요."

            return sentence 
