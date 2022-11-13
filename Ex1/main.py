import abc
import datetime
import pickle as pkl

import numpy as np
import spacy
from datasets import load_dataset
from spacy.attrs import LEMMA, IS_ALPHA
from spacy.tokens import Doc, Token

UNIGRAM_PKL = "unigram.pickle"
BIGRAM_PKL = "bigram.pickle"

START = 'START'
COUNTER_KEY = '#'
ATTR_LIST = [LEMMA, IS_ALPHA]

nlp = spacy.load("en_core_web_sm")
# nlp = spacy.load("en")

class NGram:
    def __init__(self):
        self.dictionary = {COUNTER_KEY: 0}

    def train(self, dataset):
        for text in dataset['text']:

            doc = nlp(text)
            filtered_doc = Doc(doc.vocab, words=[START]+[t.lemma_ for t in doc if t.is_alpha])
            for i, token in enumerate(filtered_doc):
                if token.text == START:
                    continue
                self._handle_token(filtered_doc[i - 1], token)

    def continue_sentence(self, sentence: str):
        doc = nlp(sentence)
        filtered_doc = Doc(doc.vocab, words=[START] + [t.lemma_ for t in doc if t.is_alpha])
        return self._continue_sentence(filtered_doc)

    def probability_of_sentence(self, sentence: str):
        doc = nlp(sentence)
        filtered_doc = Doc(doc.vocab, words=[START] + [t.lemma_ for t in doc if t.is_alpha])
        p = 0
        for i, word in enumerate(filtered_doc):
            if i == 0:
                continue
            p += self.probability_of_word(filtered_doc, i)
        return p

    def perplexity(self, sentences: list):
        token_set = []
        probability = 0
        for i, sentence in enumerate(sentences):
            probability += self.probability_of_sentence(sentence) * np.log2(np.e)
            token_set += [t.lemma_ for t in nlp(sentence) if t.is_alpha]

        M = len(token_set)
        l = probability / M
        return np.power(2, -l)

    @staticmethod
    def _update_dictionary(dictionary: dict, key: str, default=0):
        dictionary[key] = dictionary.get(key, default) + 1
        dictionary[COUNTER_KEY] = dictionary.get(COUNTER_KEY) + 1

    @abc.abstractmethod
    def _handle_token(self, *args):
        pass

    @abc.abstractmethod
    def probability_of_word(self, sentence: Doc, i: int):
        pass

    @abc.abstractmethod
    def _continue_sentence(self, sentence: Doc):
        pass

    @staticmethod
    def _get_lemma(token: Token):
        return token.lemma_ if token.lemma and token.text != START else token.text

    @staticmethod
    def _max_value_from_dictionary(dictionary: dict):
        return max(dictionary, key=lambda k: dictionary[k] if k != COUNTER_KEY else 0)


class UniGram(NGram):
    def _handle_token(self, prev: Token, token: Token):
        NGram._update_dictionary(self.dictionary, NGram._get_lemma(token))

    def _continue_sentence(self, sentence: Doc):
        return NGram._max_value_from_dictionary(self.dictionary)

    def probability_of_word(self, sentence: Doc, i: int):
        this_token = sentence[i]
        prob = self.dictionary.get(NGram._get_lemma(this_token), 0) / self.dictionary.get(COUNTER_KEY, 1)
        return np.log(prob)


class BiGram(NGram):
    def _handle_token(self, prev: Token, token: Token):
        first_word_dict = self.dictionary.get(NGram._get_lemma(prev), {COUNTER_KEY: 0})
        NGram._update_dictionary(first_word_dict, NGram._get_lemma(token))
        self.dictionary[NGram._get_lemma(prev)] = first_word_dict

    def _continue_sentence(self, sentence: Doc):
        last_token = sentence[len(sentence) - 1]
        return NGram._max_value_from_dictionary(self.dictionary[NGram._get_lemma(last_token)])

    def probability_of_word(self, sentence: Doc, i: int):
        last_token = NGram._get_lemma(sentence[i - 1])
        this_token = NGram._get_lemma(sentence[i])
        last_dictionary = self.dictionary.get(last_token, 0)
        prob = last_dictionary.get(this_token, 0) / last_dictionary.get(COUNTER_KEY, 1)
        return -np.inf if prob == 0 else np.log(prob)


class Linear_interpolation(NGram):
    gamma_unigram = 1.0 / 3
    gamma_bigram = 1.0 - gamma_unigram

    def __init__(self, unigram: UniGram, bigram: BiGram):
        self.unigram = unigram
        self.bigram = bigram

    def probability_of_word(self, sentence: Doc, i: int):
        uni_prob = self.unigram.probability_of_word(sentence, i)
        bi_prob = self.bigram.probability_of_word(sentence, i)
        return np.log(self.gamma_unigram * np.exp(uni_prob) + self.gamma_bigram * np.exp(bi_prob))


def train_all():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    unigram = UniGram()
    bigram = BiGram()

    unigram.train(dataset)
    bigram.train(dataset)
    with open(UNIGRAM_PKL, "wb") as f:
        pkl.dump(unigram, f)
    with open(BIGRAM_PKL, "wb") as f:
        pkl.dump(bigram, f)
    return unigram, bigram


def load_all():
    unigram = UniGram()
    bigram = BiGram()

    with open(UNIGRAM_PKL, "rb") as f:
        unigram = pkl.load(f)
    with open(BIGRAM_PKL, "rb") as f:
        bigram = pkl.load(f)
    return unigram, bigram


def main():
    unigram, bigram = load_all()
    print(f'unigram: {unigram.continue_sentence("I have a house in")}')
    print(f'bigram: {bigram.continue_sentence("I have a house in")}')

    BRAD_PIT = "Brad Pitt was born in Oklahoma"
    ACTOR = "The actor was born in USA"
    print(f'\nprobability for brad pit with bigram is {bigram.probability_of_sentence(BRAD_PIT)}')
    print(f'probability for Actor with bigram is {bigram.probability_of_sentence(ACTOR)}')
    print(f'the preplexity with bigram is: {bigram.perplexity([BRAD_PIT, ACTOR])}')

    # print(f'\nprobability for brad pit with unigram is {unigram.probability_of_sentence(BRAD_PIT)}')
    # print(f'probability for Actor with unigram is {unigram.probability_of_sentence(ACTOR)}')
    # print(f'the preplexity with unigram is: {bigram.perplexity([BRAD_PIT, ACTOR])}')

    linear_inter = Linear_interpolation(unigram, bigram)
    print(f'\nprobability for brad pit with linear is {linear_inter.probability_of_sentence(BRAD_PIT)}')
    print(f'probability for Actor with linear is {linear_inter.probability_of_sentence(ACTOR)}')
    print(f'the preplexity with linear is: {linear_inter.perplexity([BRAD_PIT, ACTOR])}')


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    train_all()
    print(f'training time: {datetime.datetime.now()-start_time}\n')
    main()