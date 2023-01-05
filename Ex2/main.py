import nltk
# nltk.download('brown')

from nltk.corpus import brown as br

COUNTER_KEY = '#'  # key for the counter of the number of words in the dictionary
MOST_LIKELY_TAG = 'NN'
START = '*'
START_TAG = 'START'
STOP = '^'
STOP_TAG = 'STOP'
PLUS = '+'
MINUS = '-'


def extract_tag(tag):
    if tag.startswith(PLUS) or tag.startswith(MINUS):
        return tag
    if PLUS in tag:
        return tag.split(PLUS)[0]
    if MINUS in tag:
        return tag.split(MINUS)[0]
    return tag


class BaseTagger:
    def __init__(self):
        self.dictionary = {}

    def train(self, train_set):
        for sentence in train_set:
            for word, tag in sentence:
                tag = extract_tag(tag)
                self.tag_word(word, tag)
        return self.dictionary

    def tag_word(self, word, tag):
        tag = extract_tag(tag)
        word_dict = self.dictionary.get(word, {COUNTER_KEY: 0})
        word_dict[tag] = word_dict.get(tag, 0) + 1
        word_dict[COUNTER_KEY] += 1
        self.dictionary[word] = word_dict

    def probability_of_word(self, word, tag):
        tag = extract_tag(tag)
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
                tag = extract_tag(tag)
                if word in self.dictionary:
                    known.append((word, tag))
                else:
                    unknown.append((word, tag))
        return [known], [unknown]


class BiGramHMM:
    def __init__(self, use_smoothing=False):
        self.tag_tag_dictionary = {}
        self.word_tag_dictionary = {}
        self.use_smoothing = use_smoothing
        self.tag_set = set()
        self.word_set = set()

    def train(self, train_set):
        for sentence in train_set:
            sentence_start_stop = [(START, START_TAG)] + sentence + [(STOP, STOP_TAG)]
            for i, word_tag in enumerate(sentence_start_stop):
                word, tag = word_tag
                tag = extract_tag(tag)

                self.word_set.add(word)
                self.tag_set.add(tag)

                self.tag_word(word, tag)
                if i == 0:
                    continue
                prev_tag = sentence_start_stop[i - 1][1]
                self.tag_tag(prev_tag=prev_tag, curr_tag=tag)

    def tag_tag(self, prev_tag, curr_tag):
        prev_tag, curr_tag = extract_tag(prev_tag), extract_tag(curr_tag)
        tag_dict = self.tag_tag_dictionary.get(prev_tag, {COUNTER_KEY: 0})
        tag_dict[curr_tag] = tag_dict.get(curr_tag, 0) + 1
        tag_dict[COUNTER_KEY] += 1
        self.tag_tag_dictionary[prev_tag] = tag_dict

    def tag_word(self, word, tag):
        tag = extract_tag(tag)
        tag_dict = self.word_tag_dictionary.get(tag, {COUNTER_KEY: 0})
        tag_dict[word] = tag_dict.get(word, 0) + 1
        tag_dict[COUNTER_KEY] += 1
        self.word_tag_dictionary[tag] = tag_dict

    def emission_probability(self, word, tag):
        tag = extract_tag(tag)
        tag_dict = self.word_tag_dictionary.get(tag, {})
        if not tag_dict:
            return 0
        emission = tag_dict.get(word, 0) / tag_dict[COUNTER_KEY]
        return emission

    def transition_probability(self, prev_tag, curr_tag):
        prev_tag, curr_tag = extract_tag(prev_tag), extract_tag(curr_tag)
        tag_dict = self.tag_tag_dictionary.get(prev_tag, {})
        if not tag_dict:
            return 0
        if self.use_smoothing:
            transition = tag_dict.get(curr_tag, 0) + 1
            transition /= tag_dict[COUNTER_KEY] + len(self.word_set)
        else:
            transition = tag_dict.get(curr_tag, 0) / tag_dict[COUNTER_KEY]
        return transition


def viterbi(sentence, hmm: BiGramHMM):
    def get_pi_value(word, tag, prev_tag):
        return hmm.transition_probability(prev_tag, tag) * hmm.emission_probability(word, tag)

    S = [{START_TAG}]
    table = [{START_TAG: 1}]
    all_tags = hmm.tag_set
    for k, (word, _) in enumerate(sentence):
        S.append(all_tags)  # Sk = S
        table.append({})
        for u in S[k+1]:
            pi_values = [table[k][w] * get_pi_value(word, tag=u, prev_tag=w) for w in S[k]]
            max_value = max(pi_values)
            table[k+1][u] = max_value
    tags = []
    for k, (word, _) in enumerate(sentence):
        max_value_tag = max(all_tags, key=lambda u: table[k].get(u, 0) * hmm.transition_probability(u, STOP_TAG))
        tags.append((word, max_value_tag))
    return tags


def predict_viterbi_set(test_set, hmm: BiGramHMM):
    predict_set = []
    for sentence in test_set:
        predict_set.append(viterbi(sentence, hmm))
    return predict_set


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
    predicted_known = predict_viterbi_set(known, bigram)
    predicted_unknown = predict_viterbi_set(unknown, bigram)
    predicted = predict_viterbi_set(test, bigram)
    print('Error rate on known words viterbi: {}'.format(1 - accuracy(known, predicted_known)))
    print('Error rate on unknown words viterbi: {}'.format(1 - accuracy(unknown, predicted_unknown)))
    print('Error rate all words viterbi: {}'.format(1 - accuracy(test, predicted)))

    bigram = BiGramHMM(use_smoothing=True)
    bigram.train(train)
    predicted_known = predict_viterbi_set(known, bigram)
    predicted_unknown = predict_viterbi_set(unknown, bigram)
    predicted = predict_viterbi_set(test, bigram)
    print('Error rate on known words viterbi and add one: {}'.format(1 - accuracy(known, predicted_known)))
    print('Error rate on unknown words viterbi and add one: {}'.format(1 - accuracy(unknown, predicted_unknown)))
    print('Error rate all words viterbi and add one: {}'.format(1 - accuracy(test, predicted)))


def get_section_from_corpus(section):
    return br.tagged_sents(categories=section)


if __name__ == '__main__':
    main()
