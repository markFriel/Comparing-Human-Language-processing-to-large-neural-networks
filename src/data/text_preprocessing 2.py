import re

import numpy as np


class preprocessing_text():

    def clean_text(self, word_list):
        """ utilises the other methods in the class to perform text cleaning in one function call

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        words : list
            list of words with puntuation removed

        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        words = self.remove_puntuation(word_list)
        words = self.words_to_lower(words)
        words = self.remove_non_ascii(words)
        words = self.remove_empty_space(words)

        return words

    def remove_puntuation(self, word_list):
        """ Removes puntuation from the word in the word_list

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        words : list
            list of words with puntuation removed
        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        words = [re.sub('[?!.,"#$%\'()*+-/:;<=>@^_`{|}~]', '', x) for x in word_list]

        return words

    def remove_empty_space(self, word_list):
        """ Removes empty space and maps empty strings to the unknown token

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        words : list
            list of words with empty space removed
        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        new_words = []
        words = [x.replace(" ", '') for x in word_list]

        for x in words:
            if (len(x) >= 1):
                new_words.append(x)
            else:
                new_words.append('<unk>')

        return new_words

    def words_to_lower(self, word_list):
        """ converts the words in word_list to lower case

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        words : list
            list of words with upper case letters converted to lowere case
        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        words = [x.lower() for x in word_list]

        return words

    def remove_non_ascii(self, word_list):
        """ Removes non-ascii characters from words in word list

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        words : list
            list of words with non-ascii charcaters removed
        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        for i in range(0, len(word_list)):
            word_list[i] = "".join(j for j in word_list[i] if 31 < ord(j) < 127)
        return word_list

    def word_list_to_string(self, word_list):
        """ converts the words in the word_list into a string

        Parameters
        ----------
        word_list : list
            list of the words that correspon to the neural/ eye tracking responses

        Returns
        -------
        text : string
            string of all the words in the word_list seperated by " "
        """

        if isinstance(word_list, list) != True:
            raise TypeError

        if np.asarray(word_list).ndim != 1:
            raise ValueError

        text = ' '.join(word_list)

        return text
