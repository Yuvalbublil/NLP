import nltk
# nltk.download('brown')
from nltk.corpus import brown as br

COUNTER_KEY = '#'  # key for the counter of the number of words in the dictionary
MOST_LIKELY_TAG = 'NN'
START = '*'
STOP = '^'


class BaseTagger:
    def __init__(self):
        self.dictionary = {}

    def train(self, train_set):
        for sentence in train_set:
            for word, tag in sentence:
                self.tag_word(word, tag)
        return self.dictionary

    def tag_word(self, word, tag):
        word_dict = self.dictionary.get(word, {COUNTER_KEY: 0})
        word_dict[tag] = word_dict.get(tag, 0) + 1
        word_dict[COUNTER_KEY] += 1
        self.dictionary[word] = word_dict

    def probability_of_word(self, word, tag):
        word_dict = self.dictionary.get(word, {})
        if not word_dict:
            return 0
        return word_dict.get(tag, 0) / word_dict[COUNTER_KEY]

    def predict_word(self, word):
        word_dict = self.dictionary.get(word, {})
        if not word_dict:
            return MOST_LIKELY_TAG
        return max(word_dict, key=lambda k: word_dict[k] if not k == COUNTER_KEY else 0)

    def predict_sentence(self, sentence):
        return [(word, self.predict_word(word)) for word, tag in sentence]

    def predict_set(self, test_set):
        return [self.predict_sentence(sentence) for sentence in test_set]

    def split_known_unknown(self, test_set):
        known = []
        unknown = []
        for sentence in test_set:
            for word, tag in sentence:
                if word in self.dictionary:
                    known.append((word, tag))
                else:
                    unknown.append((word, tag))
        return [known], [unknown]


class BiGramHMM:
    def __init__(self):
        self.tag_tag_dictionary = {}
        self.word_tag_dictionary = {}

    def train(self, train_set):
        for sentence in train_set:
            sentence_start_stop = [(START, START)] + sentence + [(STOP, STOP)]
            for i, word_tag in enumerate(sentence_start_stop):
                word, tag = word_tag
                self.tag_word(word, tag)
                if i == 0:
                    continue
                prev_tag = sentence_start_stop[i - 1][1]
                self.tag_tag(prev_tag=prev_tag, curr_tag=tag)

    def tag_tag(self, prev_tag, curr_tag):
        tag_dict = self.tag_tag_dictionary.get(prev_tag, {COUNTER_KEY: 0})
        tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1
        tag_dict[COUNTER_KEY] += 1
        self.tag_tag_dictionary[prev_tag] = tag_dict

    def tag_word(self, word, tag):
        tag_dict = self.word_tag_dictionary.get(tag, {COUNTER_KEY: 0})
        tag_dict[word] = tag_dict.get(word, 0) + 1
        tag_dict[COUNTER_KEY] += 1
        self.word_tag_dictionary[tag] = tag_dict

    def emission_probability(self, word, tag):
        tag_dict = self.word_tag_dictionary.get(tag, {})
        if not tag_dict:
            return 0
        return tag_dict.get(word, 0) / tag_dict[COUNTER_KEY]

    def transition_probability(self, prev_tag, curr_tag):
        tag_dict = self.tag_tag_dictionary.get(prev_tag, {})
        if not tag_dict:
            return 0
        return tag_dict.get(curr_tag, 0) / tag_dict[COUNTER_KEY]

    def get_all_tags(self):
        return set(self.tag_tag_dictionary.keys())


def viterbi(sentence, hmm: BiGramHMM):
    def get_pi_value(word, tag, prev_tag):
        return table[prev_tag] * hmm.transition_probability(prev_tag, tag)\
               * hmm.emission_probability(word, tag)
    n = len(sentence)
    S = [{START}]
    table = {START: 1}
    for k in range(n):
        word, _ = sentence[k]
        S.append(hmm.get_all_tags())  # Sk = S
        for u in S[k+1]:
            pi_values = [get_pi_value(word, tag=u, prev_tag=w) for w in S[k]]
            max_value = max(pi_values)
            table[u] = max_value
    return max(table[u] * hmm.transition_probability(u, STOP) for u in S[n])


    # n = len(sentence)
    # tags = list(hmm.word_tag_dictionary.keys())
    # v = np.zeros((n, len(tags)))
    # b = np.zeros((n, len(tags)))
    # for i, word in enumerate(sentence):
    #     for j, tag in enumerate(tags):
    #         if i == 0:
    #             v[i, j] = hmm.emission_probability(word, tag)
    #             b[i, j] = 0
    #         else:
    #             v[i, j] = max([v[i - 1, k] * hmm.transition_probability(tags[k], tag) * hmm.emission_probability(word, tag) for k in
    #                            range(len(tags))])
    #             b[i, j] = np.argmax([v[i - 1, k] * hmm.transition_probability(tags[k], tag) for k in range(len(tags))])
    # tags = []
    # i = np.argmax(v[n - 1, :])
    # tags.append(i)
    # for j in range(n - 1, 0, -1):
    #     i = int(b[j, int(i)])
    #     tags.append(i)
    # tags.reverse()
    # return [(word, tags[i]) for i, word in enumerate(sentence)]


def accuracy(test_set, predicted_set):
    correct = 0
    total = 0
    for test_sentence, predicted_sentence in zip(test_set, predicted_set):
        for test, predicted in zip(test_sentence, predicted_sentence):
            test_word, test_tag = test
            predicted_word, predicted_tag = predicted
            if test_word == predicted_word:
                total += 1
                if test_tag == predicted_tag:
                    correct += 1
            else:
                raise Exception('Test and predicted words do not match!')
    return correct / total


def main():
    news = get_section_from_corpus('news')
    split = int(len(news) * 0.9)
    train = news[:split]
    test = news[split:]
    base_tagger = BaseTagger()
    base_tagger.train(train)
    known, unknown = base_tagger.split_known_unknown(test)
    predicted_known = base_tagger.predict_set(known)
    predicted_unknown = base_tagger.predict_set(unknown)
    predicted = base_tagger.predict_set(test)
    print('Error rate all words: {}'.format(1 - accuracy(test, predicted)))
    print('Error rate on known words: {}'.format(1 - accuracy(known, predicted_known)))
    print('Error rate on unknown words: {}'.format(1 - accuracy(unknown, predicted_unknown)))

    bigram = BiGramHMM()
    bigram.train(train)
    print(f"Viterbi result is {viterbi(train[0], bigram)}")


def get_section_from_corpus(section):
    return br.tagged_sents(categories=section)


if __name__ == '__main__':
    main()
