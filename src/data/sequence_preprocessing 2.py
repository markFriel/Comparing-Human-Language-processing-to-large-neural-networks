from collections import Counter
import warnings
import mne.evoked
import numpy as np
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from scipy.stats import pearsonr
from transformers import GPT2Tokenizer, TransfoXLTokenizer, CTRLTokenizer


class sequence_processor():

    def __init__(self, modelType, text):

        if (modelType == None or text == None):
            raise TypeError

        if (type(modelType) != str or type(text) != str):
            raise TypeError

        self.tokenizer = self._loadTokenizer(modelType)
        self.text = text

    def _loadTokenizer(self, modelType):
        """Loads a tokenizer from one of the Languge models based on the modelType parameter

        Parameters
        ----------
        modelType : string
            String specifying the language model that the tokenizer will be loaded

            .. versionadded:: 0.0.0
        """

        if (modelType == 'gpt2'):
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2', unk_token='<unk>')
            return tokenizer
        elif (modelType == 'gpt2-xl'):
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl', unk_token='<unk>')
            return tokenizer
        elif (modelType == 'txl'):
            tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103', unk_token='<unk>')
            return tokenizer
        elif (modelType == 'ctrl'):
            tokenizer = CTRLTokenizer.from_pretrained('ctrl', unk_token='<unk>')
            return tokenizer
        else:
            print('\n')
            raise NameError(
                ' The model parameter passed is not currently supported. Language Models currently supported are \n'
                'GPT2, GPT2 XL,  Transformer XL, and CTRL. The codes for these model are [\'gpt2\', \'gpt2-xl\', \'txl\' and \'ctrl\'] '
                'resecptivly')

    def tokenizeWords(self):
        """Tokenizes the text, each word in the text is converted to an integer that maps to a dictionary of word """

        self.indexed_tokens = self.tokenizer.encode(self.text)

    def text_to_sequences(self, contextLength):
        """Creates sequences of indexed token from the self.indexed_tokens

        Parameters
        ----------
        contextLength: int
            specifies how long the sequences are

        Notes
        -----
        sequence_processor variable indexed_sequeces is set
        sequence_processor variable indexed_nextWord is set

        .. versionadded:: 0.0.0

        """

        if (contextLength > len(self.text)):
            warnings.warn('The context length passed is longer than the text length, no sequence will\n'
                          'match the length of context specified')

        number_of_sequences = len(self.indexed_tokens)

        sequences = [None] * (number_of_sequences - 1)
        nextWord = [None] * (number_of_sequences - 1)

        for i in range(number_of_sequences - 1):
            if (i <= contextLength):
                sequences[i] = self.indexed_tokens[0:i + 1]
                nextWord[i] = self.indexed_tokens[i + 1]
            else:
                sequences[i] = self.indexed_tokens[i - contextLength:i + 1]
                nextWord[i] = self.indexed_tokens[i + 1]

        self.indexed_sequences = sequences
        self.indexed_nextWord = nextWord

    def text_to_word_sequences(self, context_length):
        """Creates sequences of words from the self.text

        Parameters
        ----------
        contextLength: int
            specifies how long the sequences are

        Notes
        -----
        sequence_processor variable indexed_sequeces is set
        sequence_processor variable indexed_nextWord is set

        .. versionadded:: 0.0.0
        """

        if (context_length > len(self.text)):
            warnings.warn('The context length passed is longer than the text length, no sequence will\n'
                          'match the length of context specified')

        word_list = self.text.split(' ')

        number_of_sequences = len(word_list)
        sequences = [None] * (number_of_sequences - 1)
        nextWord = [None] * (number_of_sequences - 1)

        for i in range(number_of_sequences - 1):
            if (i <= context_length):
                sequences[i] = word_list[0:i + 1]
                nextWord[i] = word_list[i + 1]
            else:
                sequences[i] = word_list[i - context_length:i + 1]
                nextWord[i] = word_list[i + 1]

        self.word_sequences = sequences
        self.word_nextWord = nextWord

    def word_frequency_length(self):
        """ Calculates the length and frequency of each word in self.text
        Notes
        -----
        sequence_processor variable word_lengths is set
        sequence_processor variable word_frequencies is set

        .. versionadded:: 0.0.0
        """

        word_list = self.text.split(' ')
        counter = Counter
        word_frequency = counter(word_list)

        self.word_lengths = [len(x) for x in word_list]
        self.word_frequencies = [word_frequency[x] for x in word_list]


    # TODO Fix for condition when the no word in the sequence is in the embedding file
    def calc_semantic_dissimilarity(self, file):
        """ claculates the sematic dissimilarity of a glove embedding from the previous X glove
            embeddings. X is determined by the context length which sets the sequences length
        Parameters
        ----------
        file: sting
            path to a file that contains word embeddings eg glove, word2vec

        Notes
        -----
        sequence_processor variable semantic_dissimilarity is set

        .. versionadded:: 0.0.0
        """

        word2vec_output_file = file + 'word2vec'
        glove2word2vec(file, word2vec_output_file)
        filename = file + 'word2vec'
        glove = KeyedVectors.load_word2vec_format(filename, binary=False, limit=100000)

        # calculate unknown embedding by taking average of all embeddings
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                pass
        number_of_embeddings = i + 1
        embedding_dimension = len(line.split(' ')) - 1

        all_embeddings_vector = np.zeros((number_of_embeddings, embedding_dimension), dtype=np.float32)

        with open(file, 'r') as f:
            for i, line in enumerate(f):
                all_embeddings_vector[i] = np.array([float(n) for n in line.split(' ')[1:]], dtype=np.float32)

        unknown_token = np.mean(all_embeddings_vector, axis=0)

        # Calculate the average embedding for each sequence
        average_embedding = []
        for sequence in self.word_sequences:
            word_embeddings = []
            for word in sequence:
                if word in glove.vocab:
                    word_embeddings.append(glove[word])

            average_embedding.append(np.sum(word_embeddings, axis=0) / len(word_embeddings))

        # Calculate the disimilarity by 1 minus the pearson correlation of a word and the
        # preceding sequence
        semantic_dissim = []
        for word in range(0, len(self.word_nextWord)):
            if self.word_nextWord[word] in glove.vocab:
                dissim = 1 - pearsonr(glove[self.word_nextWord[word]], average_embedding[word])[0]
                semantic_dissim.append(dissim)
            else:
                dissim = 1 - pearsonr(unknown_token, average_embedding[word])[0]
                semantic_dissim.append(dissim)

        semantic_dissim.insert(0, 0)
        self.Semantic_disimilarity = semantic_dissim
