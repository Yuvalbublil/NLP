import random
from pprint import pprint

from nltk.corpus import dependency_treebank

TAG = 'tag'

WORD = 'word'

ROOT = '<ROOT>'


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


def get_all_tags(sentences):
    T = set()
    for sentence in sentences:
        for word in sentence.nodes.values():
            if word['tag'] is None:
                continue
            T.add(word['tag'])
    T.add(ROOT)
    return T


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
    word_arcs, tag_arcs,  = {}, {}
    for i, sentence in enumerate(sentences):
        word_arcs[i] = update_sentence_arcs(sentence, field=WORD)
        tag_arcs[i] = update_sentence_arcs(sentence, field=TAG)
    return word_arcs, tag_arcs


def main():
    sentences = dependency_treebank.parsed_sents()
    train, test = train_test_split(sentences, split_percentage=0.1)

    V = get_all_words(sentences)
    T = get_all_tags(sentences)
    print(len(V)**2+ len(T)**2)
    # word_arcs, tag_arcs = get_all_arcs(sentences)
    # for i, word in word_arcs.items():
    #     print(word)
    #     print(tag_arcs[i])
    #     break

if __name__ == '__main__':
    main()
