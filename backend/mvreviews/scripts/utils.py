import pickle
import numpy as np
import re
import os

def build_vocab(path):
    word_to_idx = dict()
    word_to_idx['PAD'] = 0
    word_to_idx['unknown'] = 1
    print("vocabulary is building")

    reviews = open(path, 'r').readlines()
    for review in reviews:
        sentences = review.split("</s>")
        for sentence in sentences:
            tokens = [token.strip() for token in sentence.split()]
            for token in tokens:
                if token not in word_to_idx:
                    index = len(word_to_idx)
                    word_to_idx[token] = index
    f = open("mvreviews/data/vocab.pickle", "wb")
    pickle.dump(word_to_idx, f)
    f.close()
    return word_to_idx


def s_to_dot(text):
    text = text.replace("</s>", ".")
    return text


def punct_to_s(text):
    text = re.sub("[.!?]", "</s>", text)
    return text


def load_data_and_labels(num_class, data_path, labels_path,
                         max_sentence_len=50, max_word_len=100, train=True):
    if train:
        word_encoder = build_vocab(data_path)
    else:
        f = open("mvreviews/data/vocab.pickle", "rb")
        word_encoder = pickle.load(f)

    vocab_size = len(word_encoder)
    (sentence_len, word_len) = compute_length(data_path, max_word_len=max_word_len, max_sentence_len=max_sentence_len)
    docs = []

    with open(data_path, 'r') as review_txt:
        reviews = review_txt.readlines()
        length = len(reviews)
        for i, review in enumerate(reviews):
            sentences = []
            s = review.split("</s>")
            if len(s) > max_sentence_len:
                s = s[:max_sentence_len]
            for sentence in s:
                tokens = [token.strip() for token in sentence.split()]
                encoded_sentence = []
                # read token up to given maximum length
                if len(tokens) > max_word_len:
                    tokens = tokens[:max_word_len]
                for token in tokens:
                    if token.strip() not in word_encoder:
                        encoded_sentence.append(word_encoder['unknown'])
                    else:
                        encoded_sentence.append(word_encoder[token.strip()])
                # zero padding to word level
                if len(encoded_sentence) < max_word_len:
                    encoded_sentence = encoded_sentence + [0] * (max_word_len - len(encoded_sentence))
            sentences.append(encoded_sentence)
            # zero padding to sentence level
            if len(sentences) < max_sentence_len:
                sentences = sentences + [[0] * max_word_len] * (max_sentence_len - len(sentences))
            docs.append(sentences)

    x = np.array(docs)
    # one-hot encoding
    label = open(labels_path, 'r').readlines()
    labels = []
    for i, score in enumerate(label):
        labels.append(int(score.strip()) - 1)
    # y_labels = np.eye(num_class)[labels]
    data_and_label = [x, np.array(labels), vocab_size, sentence_len]
    return data_and_label


def build_product_list(product_path):
    f = open(product_path, 'r')
    product_to_idx = dict()
    product_to_idx['unknown'] = 0
    for line in f.readlines():
        product = line.strip()
        if product not in product_to_idx:
            index = len(product_to_idx)
            product_to_idx[product] = index
    f.close()
    fw = open("mvreviews/data/product_list.pickle", 'wb')
    pickle.dump(product_to_idx, fw)
    fw.close()
    return product_to_idx


def load_product(product_path, train=True):
    if train:
        product_to_idx = build_product_list(product_path)
    else:
        f = open("mvreviews/data/product_list.pickle", 'rb')
        product_to_idx = pickle.load(f)
        f.close()
    f1 = open(product_path, 'r')
    product_ids = []
    for line in f1.readlines():
        product = line.strip()
        if product not in product_to_idx:
            idx = product_to_idx['unknown']
            product_ids.append(idx)
        else:
            product_ids.append(product_to_idx[product])
    return [np.array(product_ids), len(product_to_idx)]


def load_product_user(product_path, user_path):
    f1 = open(product_path, 'r')
    f2 = open(user_path, 'r')
    product_to_idx = {}
    user_to_idx = {}
    product_ids = []
    user_ids = []
    for line in f1.readlines():
        product = line.strip()
        if product not in product_to_idx:
            index = len(product_to_idx)
            product_to_idx[product] = index
        product_ids.append(product_to_idx[product])
    for line in f2.readlines():
        user = line.strip()
        if user not in user_to_idx:
            index = len(user_to_idx)
            user_to_idx[user] = index
        user_ids.append(user_to_idx[user])
    return [np.array(product_ids), np.array(user_ids), len(product_to_idx), len(user_to_idx)]


def compute_length(path="mvreviews/data/review/txt", max_sentence_len=50, max_word_len=200):
    f = open(path, 'r')
    reviews = f.readlines()
    sentence_len = [len(review.split("</s>")) for review in reviews]
    # clipping sentence length
    sentence_len = np.clip(sentence_len, 0, max_sentence_len)
    word_len = []
    for review in reviews:
        temp = [len(sentence.split()) for sentence in review.split("</s>")]
        clipped_len = np.clip(temp, 0, max_word_len)
        word_len.append(clipped_len)

    return np.array(sentence_len), np.array(word_len)


def batch_iter(data, batch_size, num_epoch, train=True):
    """
    generate a batch iterator for a given dataset
    :param data: zip(x_train, y_train, sent_len_train, word_len_train)
    :param batch_size: batch size
    :param num_epoch: number of iteration
    :return: batch iterator
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epoch):
        print("number of epoch :{}".format(epoch + 1))
        # Shuffle the data at each epoch
        if train:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
