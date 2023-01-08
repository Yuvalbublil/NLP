import random
from pprint import pprint
from scipy.sparse import coo_matrix
from nltk.corpus import dependency_treebank

TAG = 'tag'

WORD = 'word'

ROOT = '<ROOT>'
d = 0  # global variable for d
amount_of_tags = 0
amount_of_words = 0
tags_enumarations = {}
words_enumarations = {}

def train_test_split(sentences, split_percentage):
    random.shuffle(sentences)
    split_index = int(len(sentences) * split_percentage)
    train = sentences[split_index:]
    test = sentences[:split_index]
    return train, test


def get_all_words(sentences):
    V = set()
    for sentence in sentences:
        for word in sentence.nodes.values():
            if word['word'] is None:
                continue
            V.add(word['word'])
    V.add(ROOT)
    return V

def get_words_enumaration():
    """
    :return: a dictionary of words and their index
    """
    global words_enumarations
    if words_enumarations == {}:
        sentences = dependency_treebank.parsed_sents()
        V = get_all_words(sentences)
        words_enumarations = {word: i for i, word in enumerate(V)}
    return words_enumarations

def get_all_tags(sentences):
    T = set()
    for sentence in sentences:
        for word in sentence.nodes.values():
            if word['tag'] is None:
                continue
            T.add(word['tag'])
    T.add(ROOT)
    return T

def get_tags_enumaration():
    """
    :return: a dictionary of tags and their index
    """
    global tags_enumarations
    if tags_enumarations == {}:
        sentences = dependency_treebank.parsed_sents()
        T = get_all_tags(sentences)
        tags_enumarations = {tag: i for i, tag in enumerate(T)}
    return tags_enumarations

def sentence_address_dict(sentence):
    return {word['address']: word for word in sentence.nodes.values()}


def update_sentence_arcs(sentence, field='word'):
    arcs = []
    address_dict = sentence_address_dict(sentence)
    for word in sentence.nodes.values():
        if word['head'] is None:
            continue

        head_word = address_dict[word['head']][field]
        if head_word is None:
            head_word = ROOT
        arcs.append((head_word, word[field]))
    return arcs


def get_all_arcs(sentences):
    word_arcs, tag_arcs = {}, {}
    for i, sentence in enumerate(sentences):
        word_arcs[i] = update_sentence_arcs(sentence, field=WORD)
        tag_arcs[i] = update_sentence_arcs(sentence, field=TAG)
    return word_arcs, tag_arcs

def get_d():
    """
    computing d as num_words^2 + num_tags^2
    :return:
    """
    global d
    if d == 0:
        V = get_amount_of_words()
        T = get_amount_of_tags()
        d = V ** 2 + T ** 2
    return d


def get_amount_of_tags():
    """
    :return: the amount of tags
    """
    global amount_of_tags
    if amount_of_tags == 0:
        sentences = dependency_treebank.parsed_sents()
        T = get_all_tags(sentences)
        amount_of_tags = len(T)
    return amount_of_tags


def get_amount_of_words():
    """
    :return: the amount of words
    """
    global amount_of_words
    if amount_of_words == 0:
        sentences = dependency_treebank.parsed_sents()
        V = get_all_words(sentences)
        amount_of_words = len(V)
    return amount_of_words


def make_sparse_vector(first_tag, second_tag, first_word, second_word):
    """
    making a sparse vector of size d
    tags and then words
    :param first_tag:
    :param second_tag:
    :param first_word:
    :param second_word:
    :return:
    """
    # creating the sparse vector
    arr = coo_matrix((get_d()), dtype=bool)

    # get the index of the tags and words
    second_tag_index = get_tags_enumaration()[second_tag]
    first_tag_index = get_tags_enumaration()[first_tag]
    tag_index = first_tag_index * get_amount_of_tags() + second_tag_index

    first_word_index = get_words_enumaration()[first_word]
    second_word_index = get_words_enumaration()[second_word]
    word_index = get_amount_of_tags() ** 2 + first_word_index * get_amount_of_words() + second_word_index

    # update the sparse vector
    arr[tag_index] = True
    arr[word_index] = True
    return arr

def feature_function(sentence, i, j):
    """
    :param sentence: the sentence
    :param i: the first word
    :param j: the second word
    :return: the feature vector
    """
    # get the words and tags
    # TODO: should we add .values()?
    first_word = sentence.nodes[i][WORD]
    second_word = sentence.nodes[j][WORD]
    first_tag = sentence.nodes[i][TAG]
    second_tag = sentence.nodes[j][TAG]

    # make the sparse vector
    arr = make_sparse_vector(first_tag, second_tag, first_word, second_word)
    return arr
def main():

    sentences = dependency_treebank.parsed_sents()
    train, test = train_test_split(sentences, split_percentage=0.1)

    V = get_all_words(sentences)
    T = get_all_tags(sentences)
    print(len(V) ** 2 + len(T) ** 2)
    # word_arcs, tag_arcs = get_all_arcs(sentences)
    # for i, word in word_arcs.items():
    #     print(word)
    #     print(tag_arcs[i])
    #     break


if __name__ == '__main__':
    main()
