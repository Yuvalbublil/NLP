import abc
import pickle as pkl
import spacy
from datasets import load_dataset
from spacy.attrs import LEMMA, IS_ALPHA
from spacy.tokens import Doc, Token

START = 'START'
COUNTER_KEY = '#'
ATTR_LIST = [LEMMA, IS_ALPHA]

nlp = spacy.load("en_core_web_sm")


class NGram:
    def __init__(self):
        self.dictionary = {COUNTER_KEY: 0}

    def train(self, dataset):
        for text in dataset['text']:
            doc = nlp(f"{START} {text}")
            filtered_doc = Doc(doc.vocab, words=[t.text for t in doc if t.is_alpha])
            for i, token in enumerate(filtered_doc):
                if token.text == START:
                    continue
                self._handle_token(doc[i-1], token)

    def continue_sentence(self, sentence: str):
        doc = nlp(f"{START} {sentence}")
        filtered_doc = Doc(doc.vocab, words=[t.text for t in doc if t.is_alpha])
        return self._continue_sentence(filtered_doc)

    @staticmethod
    def _update_dictionary(dictionary: dict, key: str, default=0):
        dictionary[key] = dictionary.get(key, default) + 1
        dictionary[COUNTER_KEY] = dictionary.get(COUNTER_KEY) + 1

    @abc.abstractmethod
    def _handle_token(self, *args):
        pass

    @abc.abstractmethod
    def _continue_sentence(self, sentence: Doc):
        pass

    @staticmethod
    def _get_lemma(token: Token):
        return token.lemma_ if token.lemma_ else token.text

    @staticmethod
    def _max_value_from_dictionary(dictionary: dict):
        return max(dictionary, key=lambda k: dictionary[k] if k != COUNTER_KEY else 0)


class UniGram(NGram):
    def _handle_token(self, prev: Token, token: Token):
        NGram._update_dictionary(self.dictionary, NGram._get_lemma(token))

    def _continue_sentence(self, sentence: Doc):
        return NGram._max_value_from_dictionary(self.dictionary)


class BiGram(NGram):
    def _handle_token(self, prev: Token, token: Token):
        first_word_dict = self.dictionary.get(NGram._get_lemma(token), {COUNTER_KEY: 0})
        NGram._update_dictionary(first_word_dict, NGram._get_lemma(prev))
        self.dictionary[NGram._get_lemma(token)] = first_word_dict

    def _continue_sentence(self, sentence: Doc):
        last_token = sentence[len(sentence)-1]
        return NGram._max_value_from_dictionary(self.dictionary[NGram._get_lemma(last_token)])

def train_all():
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split="train")

    unigram = UniGram()
    bigram = BiGram()

    unigram.train(dataset)
    bigram.train(dataset)
    with open("unigram.pkl", "wb") as f:
        pkl.dump(unigram, f)
    with open("bigram.pkl", "wb") as f:
        pkl.dump(bigram, f)
    return unigram, bigram

def load_all():
    unigram = UniGram()
    bigram = BiGram()

    with open("unigram.pkl", "rb") as f:
        unigram = pkl.load(f)
    with open("bigram.pkl", "rb") as f:
        bigram = pkl.load(f)
    return unigram, bigram
def main():
    unigram, bigram = load_all()
    print(f'unigram_{unigram.continue_sentence("I have a house in")}_')
    print(f'bigram_{bigram.continue_sentence("I have a house in")}_')


if __name__ == "__main__":
    # train_all()
    main()
