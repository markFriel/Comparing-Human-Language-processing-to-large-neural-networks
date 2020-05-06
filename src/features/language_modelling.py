import math

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TransfoXLTokenizer, TransfoXLLMHeadModel, CTRLTokenizer, \
    CTRLLMHeadModel


class languageModel:

    def __init__(self, modelType):
        self.tokenizer, self.model = self.modelLoader(modelType)

    def modelLoader(self, modelType):
        """Loads a tokenizer and model from one of the Languge models based on the modelType parameter

        Parameters
        ----------
        modelType : string
            String specifying the language model that the tokenizer will be loaded

        Returns
        -------
        tokenizer:
            The tokenizer corresponding to the modelType string

        model:
            The language model type correpsonding to the modelType string

        .. versionadded:: 0.0.0
        """
        if (modelType == 'gpt2'):
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            model = GPT2LMHeadModel.from_pretrained('gpt2')
            return tokenizer, model

        elif (modelType == 'gpt2-xl'):

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
            model = GPT2LMHeadModel.from_pretrained('gpt2-xl')

            return tokenizer, model

        elif (modelType == 'txl'):
            tokenizer = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
            model = TransfoXLLMHeadModel.from_pretrained('transfo-xl-wt103')

            return tokenizer, model

        elif (modelType == 'ctrl'):
            tokenizer = CTRLTokenizer.from_pretrained('ctrl')
            model = CTRLLMHeadModel.from_pretrained('ctrl')

            return tokenizer, model

        else:
            print('\n')

            raise NameError(
                ' The model parameter passed is not currently supported. Language Models currently supported are \n'
                'GPT2, GPT2 XL,  Transformer XL, and CTRL. The codes for these model are [\'gpt2\', \'gpt2-xl\', \'txl\''
                ' and \'ctrl\'] '
                'resecptivly')

    def word_probability(self, seqs, nextWord):
        """calculates the probability the langage models assigns the the next word after a sequence

        Parameters
        ----------
        seqs : 2D list
            lists of lists represetning the sequences for the language model to use
            for calculating the probability distribution

        nextWord: list
            A list of indexed token used in the function to select the correct word
            from the models proability distribution

        Returns
        -------
        probability: float
            a logit proability value for each word in the text

        word: list
            A list of decoded words from the text

        .. versionadded:: 0.0.0
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model.to(device)

        probability = [None] * len(seqs)
        decodedText = [None] * len(seqs)

        # defining the softmax function
        sft = torch.nn.Softmax(dim=0)

        for i in range(len(seqs)):
            input_ids = torch.tensor(seqs[i]).unsqueeze(0)
            input_ids = input_ids.to(device)

            with torch.no_grad():
                output = self.model(input_ids)
                predictions = output[0]

            predicted_probabilities = sft(predictions[0, -1])
            next_wrd_prob = predicted_probabilities[nextWord[i]].item()
            decodedText[i] = self.tokenizer.decode(nextWord[i])
            probability[i] = next_wrd_prob

        self.probability = probability
        self.word = decodedText

    # TODO Use a filter to get rid of elements
    def clean_predicted_words(self, firstTokenWord):
        """Clean the word list from the decoded text from the language model
        it removes added charcaters appended by the language model

        Parameters
        ----------
        firstTokenWord : integer
            integer representing the first token in the indexed tokens

        Returns
        -------
        word: list
            A list of decoded words from the text

        .. versionadded:: 0.0.0
        """

        firstToken = self.tokenizer.decode(firstTokenWord)

        self.word.insert(0, firstToken)
        self.probability.insert(0, 1.00)
        self.word = [x.strip(' ') for x in self.word]
        self.word = [x.replace('@', '') for x in self.word]
        self.word = [x.replace(" ", "") for x in self.word]

    def collaspeSubWords(self, original_word_list):
        """Collaspes the word output from the language model so as to be the same as the
            original words passed to the tokenizer. Chain rule is used to calculate the
            probability of the complete words that are broken into sub-words

        Parameters
        ----------
        original_word_list : list
            list of the original words beforre the tokenization stage

        Returns
        -------
        word: list
            A list of decoded words from the text

        probability: list
            The probaility values of the words computed by the language model

        .. versionadded:: 0.0.0
        """


        self.word = [x.strip(' ') for x in self.word]
        out_copy = self.word.copy()
        value_copy = self.probability.copy()
        out_pos = 0
        org_pos = 0
        offset = 0

        # collasped the sub words and compute probability by chain rule
        for i in original_word_list:
            if (out_copy[out_pos] == original_word_list[org_pos]):
                out_pos += 1
            else:
                string = out_copy[out_pos]
                prob = value_copy[out_pos]
                count = 1
                while (string != original_word_list[org_pos]):
                    string += out_copy[out_pos + count]
                    prob = value_copy[out_pos + count] * prob
                    count += 1

                self.word.insert(org_pos + offset, string)
                self.probability.insert(org_pos + offset, prob)

                for j in range(count):
                    offset += 1
                    self.word[org_pos + offset] = "GETRIDOFF"
                    self.probability[org_pos + offset] = "GETRIDOFF"
                    out_pos += 1
            org_pos += 1


        # Removing those words that are marked and are now redundant
        true = [None] * len(original_word_list)
        true_val = [None] * len(original_word_list)
        position = 0
        for i in range(len(self.word)):
            if (self.word[i] != "GETRIDOFF"):
                true[position] = self.word[i]
                true_val[position] = self.probability[i]
                position += 1

        self.probability = true_val
        self.word = true


    def computeSurprisal(self):
        """ computed the surprisal of each element in a list
                surprisal = the negative log probability
        Returns:
        -------
        Surprisal: list
            Surprisal values computed from the probability values output by the language model
        """

        self.surprisal = self.probability
        for i in range(len(self.probability) - 1):
            sur = math.log(self.probability[i], 2) * -1
            self.surprisal[i] = sur
