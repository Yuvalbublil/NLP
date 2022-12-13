import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, Dataset
import operator
import data_loader
import pickle
import tqdm
import gensim
from memory_profiler import profile

# ------------------------------------------- Constants ----------------------------------------

SEQ_LEN = 52
W2V_EMBEDDING_DIM = 300

ONEHOT_AVERAGE = "onehot_average"
W2V_AVERAGE = "w2v_average"
W2V_SEQUENCE = "w2v_sequence"
PATHS = {ONEHOT_AVERAGE: "results_log_linear_one_hot",
         W2V_AVERAGE: "results_log_linear_w2v_average",
         W2V_SEQUENCE: "results_LSTM_w2v_sequence"}
TRAIN = "train"
VAL = "val"
TEST = "test"


# ------------------------------------------ Helper methods and classes --------------------------

def get_available_device():
    """
    Allows training on GPU if available. Can help with running things faster when a GPU with cuda is
    available but not a most...
    Given a device, one can use module.to(device)
    and criterion.to(device) so that all the computations will be done on the GPU.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_model(model, path, epoch, optimizer):
    """
    Utility function for saving checkpoint of a model, so training or evaluation can be executed later on.
    :param model: torch module representing the model
    :param optimizer: torch optimizer used for training the module
    :param path: path to save the checkpoint into
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()}, path)


def load(model, path, optimizer):
    """
    Loads the state (weights, paramters...) of a model which was saved with save_model
    :param model: should be the same model as the one which was saved in the path
    :param path: path to the saved checkpoint
    :param optimizer: should be the same optimizer as the one which was saved in the path
    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


# ------------------------------------------ Data utilities ----------------------------------------

def load_word2vec():
    """ Load Word2Vec Vectors
        Return:
            wv_from_bin: All 3 million embeddings, each lengh 300
    """
    import gensim.downloader as api
    wv_from_bin = api.load("word2vec-google-news-300")
    vocab = list(wv_from_bin.vocab.keys())
    print(wv_from_bin.vocab[vocab[0]])
    print("Loaded vocab size %i" % len(vocab))
    return wv_from_bin


def create_or_load_slim_w2v(words_list, cache_w2v=False) -> dict:
    """
    returns word2vec dict only for words which appear in the dataset.
    :param words_list: list of words to use for the w2v dict
    :param cache_w2v: whether to save locally the small w2v dictionary
    :return: dictionary which maps the known words to their vectors
    """
    w2v_path = "w2v_dict.pkl"
    if not os.path.exists(w2v_path):
        full_w2v = load_word2vec()
        w2v_emb_dict = {k: full_w2v[k] for k in words_list if k in full_w2v}
        if cache_w2v:
            save_pickle(w2v_emb_dict, w2v_path)
    else:
        w2v_emb_dict = load_pickle(w2v_path)
    return w2v_emb_dict


def get_w2v_average(sent, word_to_vec, embedding_dim):
    """
    This method gets a sentence and returns the average word embedding of the words consisting
    the sentence.
    :param sent: the sentence object
    :param word_to_vec: a dictionary mapping words to their vector embeddings
    :param embedding_dim: the dimension of the word embedding vectors
    :return The average embedding vector as numpy ndarray.
    """
    vec = np.zeros(embedding_dim)
    for word in sent:
        vec += word_to_vec[word][:embedding_dim]
    return vec / len(sent)


def get_one_hot(size, ind):
    """
    this method returns a one-hot vector of the given size, where the 1 is placed in the ind entry.
    :param size: the size of the vector
    :param ind: the entry index to turn to 1
    :return: numpy ndarray which represents the one-hot vector
    """
    vec = np.zeros(size)
    vec[ind] = 1
    return vec


def average_one_hots(sent: data_loader.Sentence, word_to_ind):
    """
    this method gets a sentence, and a mapping between words to indices, and returns the average
    one-hot embedding of the tokens in the sentence.
    :param sent: a sentence object.
    :param word_to_ind: a mapping between words to indices
    :return:
    """
    vec = np.zeros(len(word_to_ind))
    for word in sent.text:
        vec += get_one_hot(len(word_to_ind), word_to_ind[word])
    return vec / len(sent.text)


def get_word_to_ind(words_list) -> dict:
    """
    this function gets a list of words, and returns a mapping between
    words to their index.
    :param words_list: a list of words
    :return: the dictionary mapping words to the index
    """
    words_list = list(set(words_list))
    return {word: i for i, word in enumerate(words_list)}


def sentence_to_embedding(sent: data_loader.Sentence, word_to_vec: dict, seq_len, embedding_dim=300):
    """
    this method gets a sentence and a word to vector mapping, and returns a list containing the
    words embeddings of the tokens in the sentence.
    :param sent: a sentence object
    :param word_to_vec: a word to vector mapping.
    :param seq_len: the fixed length for which the sentence will be mapped to.
    :param embedding_dim: the dimension of the w2v embedding
    :return: numpy ndarray of shape (seq_len, embedding_dim) with the representation of the sentence
    """
    vec = np.zeros((seq_len, embedding_dim))
    for i, word in enumerate(sent.text):
        if i >= seq_len:
            break
        vec[i] = word_to_vec[word][:embedding_dim]
    return vec


class OnlineDataset(Dataset):
    """
    A pytorch dataset which generates model inputs on the fly from sentences of SentimentTreeBank
    """

    def __init__(self, sent_data, sent_func, sent_func_kwargs):
        """
        :param sent_data: list of sentences from SentimentTreeBank
        :param sent_func: Function which converts a sentence to an input datapoint
        :param sent_func_kwargs: fixed keyword arguments for the state_func
        """
        self.data = sent_data
        self.sent_func = sent_func
        self.sent_func_kwargs = sent_func_kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sent = self.data[idx]
        sent_emb = self.sent_func(sent, **self.sent_func_kwargs)
        sent_label = sent.sentiment_class
        return sent_emb, sent_label


class DataManager():
    """
    Utility class for handling all data management task. Can be used to get iterators for training and
    evaluation.
    """

    def __init__(self, data_type=ONEHOT_AVERAGE, use_sub_phrases=True, dataset_path="stanfordSentimentTreebank",
                 batch_size=50,
                 embedding_dim=None):
        """
        builds the data manager used for training and evaluation.
        :param data_type: one of ONEHOT_AVERAGE, W2V_AVERAGE and W2V_SEQUENCE
        :param use_sub_phrases: if true, training data will include all sub-phrases plus the full sentences
        :param dataset_path: path to the dataset directory
        :param batch_size: number of examples per batch
        :param embedding_dim: relevant only for the W2V data types.
        """

        # load the dataset
        self.sentiment_dataset = data_loader.SentimentTreeBank(dataset_path, split_words=True)
        # map data splits to sentences lists
        self.sentences = {}
        if use_sub_phrases:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set_phrases()
        else:
            self.sentences[TRAIN] = self.sentiment_dataset.get_train_set()

        self.sentences[VAL] = self.sentiment_dataset.get_validation_set()
        self.sentences[TEST] = self.sentiment_dataset.get_test_set()

        # map data splits to sentence input preperation functions
        words_list = list(self.sentiment_dataset.get_word_counts().keys())
        if data_type == ONEHOT_AVERAGE:
            self.sent_func = average_one_hots
            self.sent_func_kwargs = {"word_to_ind": get_word_to_ind(words_list)}
        elif data_type == W2V_SEQUENCE:
            self.sent_func = sentence_to_embedding

            self.sent_func_kwargs = {"seq_len": SEQ_LEN,
                                     "word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        elif data_type == W2V_AVERAGE:
            self.sent_func = get_w2v_average
            words_list = list(self.sentiment_dataset.get_word_counts().keys())
            self.sent_func_kwargs = {"word_to_vec": create_or_load_slim_w2v(words_list),
                                     "embedding_dim": embedding_dim
                                     }
        else:
            raise ValueError("invalid data_type: {}".format(data_type))
        # map data splits to torch datasets and iterators
        self.torch_datasets = {k: OnlineDataset(sentences, self.sent_func, self.sent_func_kwargs) for
                               k, sentences in self.sentences.items()}
        self.torch_iterators = {k: DataLoader(dataset, batch_size=batch_size, shuffle=k == TRAIN)
                                for k, dataset in self.torch_datasets.items()}

    def get_torch_iterator(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: torch batches iterator for this part of the datset
        """
        return self.torch_iterators[data_subset]

    def get_labels(self, data_subset=TRAIN):
        """
        :param data_subset: one of TRAIN VAL and TEST
        :return: numpy array with the labels of the requested part of the datset in the same order of the
        examples.
        """
        return np.array([sent.sentiment_class for sent in self.sentences[data_subset]])

    def get_input_shape(self):
        """
        :return: the shape of a single example from this dataset (only of x, ignoring y the label).
        """
        return self.torch_datasets[TRAIN][0][0].shape


# ------------------------------------ Models ----------------------------------------------------

class LSTM(nn.Module):
    """
    An LSTM for sentiment analysis with architecture as described in the exercise description.
    """

    def __init__(self, embedding_dim, hidden_dim, n_layers, dropout):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, text):
        """
        :param text: a tensor batch of sentences. shape: (batch_size, seq_len, embedding_dim)
        :return:
        """
        return self.lstm.forward(text)

    def predict(self, text):
        return self.forward(text)[0]


class LogLinear(nn.Module):
    """
    general class for the log-linear models for sentiment analysis.
    """

    def __init__(self, embedding_dim):
        super(LogLinear, self).__init__()
        self.linear = nn.Linear(embedding_dim, 1, dtype=torch.float64)

    def forward(self, x):
        return self.linear.forward(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


# ------------------------- training functions -------------


def binary_accuracy(preds, y):
    """
    This method returns tha accuracy of the predictions, relative to the labels.
    You can choose whether to use numpy arrays or tensors here.
    :param preds: a vector of predictions
    :param y: a vector of true labels
    :return: scalar value - (<number of accurate predictions> / <number of examples>)
    """
    return torch.sum(preds == y).item() / y.size(0)


def train_epoch(model, data_iterator, optimizer, criterion: nn.BCEWithLogitsLoss):
    """
    This method operates one epoch (pass over the whole train set) of training of the given model,
    and returns the accuracy and loss for this epoch
    :param model: the model we're currently training
    :param data_iterator: an iterator, iterating over the training data for the model.
    :param optimizer: the optimizer object for the training process.
    :param criterion: the criterion object for the training process.
    """
    for batch in tqdm.tqdm(data_iterator):
        x, y = batch
        optimizer.zero_grad()
        y_pred = model(x).squeeze()
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


@profile
def evaluate(model: nn.Module, data_iterator: DataLoader, criterion):
    """
    evaluate the model performance on the given data
    :param model: one of our models..
    :param data_iterator: torch data iterator for the relevant subset
    :param criterion: the loss criterion used for evaluation
    :return: tuple of (average loss over all examples, average accuracy over all examples)
    """
    loss = 0
    accuracy = 0
    counter = 0  # TODO this function is taking too much memory, try to reduce it
    torch.no_grad()
    for batch in data_iterator:
        counter += 1
        accuracy, loss = evaluate_helper(accuracy, batch, criterion, loss, model)
    torch.enable_grad()
    return loss / counter, accuracy / counter


def evaluate_helper(accuracy, batch, criterion, loss, model):
    x, y = batch
    y_pred = model(x).squeeze()
    loss += criterion(y_pred, y).item()
    accuracy += binary_accuracy(y_pred, y)
    return accuracy, loss


def get_predictions_for_data(model, data_iter: DataLoader):
    """

    This function should iterate over all batches of examples from data_iter and return all of the models
    predictions as a numpy ndarray or torch tensor (or list if you prefer). the prediction should be in the
    same order of the examples returned by data_iter.
    :param model: one of the models you implemented in the exercise
    :param data_iter: torch iterator as given by the DataManager
    :return:
    """
    y_pred = []
    for batch in data_iter:
        x, _ = batch
        y_pred.extend(model(x))
    return y_pred


def train_model(model, data_manager: DataManager, n_epochs, lr, weight_decay=0.):
    """
    Runs the full training procedure for the given model. The optimization should be done using the Adam
    optimizer with all parameters but learning rate and weight decay set to default.
    :param model: module of one of the models implemented in the exercise
    :param data_manager: the DataManager object
    :param n_epochs: number of times to go over the whole training set
    :param lr: learning rate to be used for optimization
    :param weight_decay: parameter for l2 regularization
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(n_epochs):
        train_epoch(model, data_manager.get_torch_iterator(TRAIN), optimizer, criterion)


def train_and_evaluate(model, data_manager: DataManager, n_epochs, lr, weight_decay, subsets_loss, subsets_acc):
    train_model(model, data_manager, n_epochs, lr, weight_decay)

    for subset in [TRAIN, VAL, TEST]:
        loss, accuracy = evaluate(model, data_manager.get_torch_iterator(subset), nn.BCEWithLogitsLoss())
        subsets_loss[subset].append(loss)
        subsets_acc[subset].append(accuracy)


@profile
def train_log_linear(data_type=ONEHOT_AVERAGE, evaluate_on_test=True):
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    results_dir = PATHS[data_type]
    plot_name = f"Log Linear {data_type} Model"
    trained_model, train_loss, val_loss = pickle_handler_load(results_dir)

    if trained_model:
        if evaluate_on_test:
            plot_evaluation(train_loss, val_loss, plot_name, results_dir)
        return

    data_manager = DataManager(data_type=data_type, batch_size=64)
    embedding_dim = data_manager.get_input_shape()[0]
    model = LogLinear(embedding_dim=embedding_dim)

    n_epochs, lr, weight_decay = 1, 0.01, 0.001
    subsets_loss = {TRAIN: [], VAL: [], TEST: []}
    subsets_acc = {TRAIN: [], VAL: [], TEST: []}

    for epoch in range(1):
        if evaluate_on_test:
            train_and_evaluate(model, data_manager, n_epochs, lr, weight_decay, subsets_loss, subsets_acc)
        else:
            train_model(model, data_manager, n_epochs, lr, weight_decay)

    if evaluate_on_test:
        plot_evaluation(subsets_loss[TRAIN], subsets_loss[VAL], f"{plot_name} - Loss", PATHS[data_type])
        plot_evaluation(subsets_acc[TRAIN], subsets_acc[VAL], f"{plot_name} - Accuracy", PATHS[data_type])
    pickle_handler_save(model, subsets_loss[TRAIN], subsets_loss[VAL], PATHS[data_type])
    return model


def pickle_handler_load(results_dir):
    if not os.path.exists(results_dir):
        return None, None, None
    trained_model = load_pickle(f"{results_dir}/model.pkl")
    train_loss, val_loss = load_pickle(f"{results_dir}/loss.pkl")
    return trained_model, train_loss, val_loss


def pickle_handler_save(results_dir, model, train_loss, val_loss):
    os.mkdir(results_dir)
    if not os.path.exists(results_dir):
        dirname = os.path.dirname(__file__)
        abs_results_dir = os.path.join(dirname, results_dir)
        os.mkdir(abs_results_dir)
        return None, None, None
    save_pickle(f"{results_dir}/model.pkl", model)
    save_pickle(f"{results_dir}/loss.pkl", (train_loss, val_loss))


def plot_evaluation(train_loss_arr, val_loss_arr, model_name, results_dir):
    plt.plot(train_loss_arr, label='train loss')
    plt.plot(val_loss_arr, label='validation loss')
    plt.legend()
    plt.title(f'{model_name} loss: train vs validation')
    plt.xlabel('epoch number')
    plt.ylabel('loss')
    plt.xticks(range(len(train_loss_arr)))
    plt.savefig(f'{results_dir}/{model_name}_loss.png')
    plt.show()


def train_log_linear_with_one_hot():
    """
    Here comes your code for training and evaluation of the log linear model with one hot representation.
    """
    return train_log_linear(data_type=ONEHOT_AVERAGE)


def train_log_linear_with_w2v():
    """
    Here comes your code for training and evaluation of the log linear model with word embeddings
    representation.
    """
    return train_log_linear(data_type=W2V_AVERAGE)


def train_lstm_with_w2v():
    """
    Here comes your code for training and evaluation of the LSTM model.
    """
    return


if __name__ == '__main__':
    train_log_linear_with_one_hot()
    # train_log_linear_with_w2v()
    # train_lstm_with_w2v()
