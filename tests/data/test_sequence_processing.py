import pytest
from src.data.sequence_preprocessing import sequence_processor
from transformers import GPT2Tokenizer, TransfoXLTokenizer, CTRLTokenizer
import numpy as np
import pandas as pd

class TestSequenceProcessorInit():

    def test_sequence_processor_non_text_parameter(self):
        text = 3
        model_type='gpt2'
        with pytest.raises(TypeError):
            sequence_processor(model_type, text)


    def test_sequence_processor_empty_parameter(self):
        text = 'This is a sample text string for testing purposes'
        with pytest.raises(TypeError):
            sequence_processor(text)

class TestLoadTokenizer():

    def test_tokenizer_loads_gpt2(self):
        test_case ='gpt2'
        text ='This is a sample text string for testing purposes'
        actual = sequence_processor(test_case, text).tokenizer
        expected = GPT2Tokenizer.from_pretrained(test_case)
        assert type(actual) == type(expected)

    def test_tokenizer_loads_ctrl(self):
        test_case ='ctrl'
        text ='This is a sample text string for testing purposes'
        actual = sequence_processor(test_case, text).tokenizer
        expected = CTRLTokenizer.from_pretrained(test_case)
        assert type(actual) == type(expected)

    def test_tokenizer_loads_txl(self):
        test_case ='txl'
        text ='This is a sample text string for testing purposes'
        actual = sequence_processor(test_case, text).tokenizer
        expected = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        assert type(actual) == type(expected)

    def test_tokenizer_loads_gpt2xl(self):
        test_case ='gpt2-xl'
        text ='This is a sample text string for testing purposes'
        actual = sequence_processor(test_case, text).tokenizer
        expected = GPT2Tokenizer.from_pretrained(test_case)
        assert type(actual) == type(expected)


    def test_tokenizer_empty_incorrect_model_name(self):
        test_case ='test'
        text = 'This is a sample text string for testing purposes'
        with pytest.raises(NameError):
            sequence_processor(test_case,text)

class TestTokenizeWords():

    def test_tokenized_words_is_list(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        actual = sp.indexed_tokens
        assert type(actual) is list

    def test_indexed_tokens_is_integer(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        actual = sp.indexed_tokens[0]
        assert type(actual) is int

    def test_indexed_tokens_equals_length(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        expected = len(text.split(' '))
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        actual = len(sp.indexed_tokens)
        assert actual == expected

    def test_indexed_tokens_assigns_same_token_to_same_word(self):
        test_case = 'ctrl'
        text = 'identical identical identical identical identical identical identical'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        actual = len(np.unique(sp.indexed_tokens))
        expected =1
        assert actual == expected

class TestTextToSequence():

    def test_next_word_is_last_word_in_previous_sequence(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        last_token_in_sequence = sp.indexed_sequences[1][-1]
        print(sp.indexed_sequences)
        next_word = sp.indexed_nextWord[0]
        assert last_token_in_sequence == next_word

    def test_number_of_sequences_equals_number_nextWord(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        sequence_length = len(sp.indexed_sequences)
        nextword_length = len(sp.indexed_nextWord)
        assert  sequence_length == nextword_length


    def test_next_word_is_one_less_than_text_string(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        expected = len(text.split(' ')) - 1
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        actual = len(sp.indexed_nextWord)
        assert actual == expected

    def test_sequence_tokens_are_int(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        actual = type(sp.indexed_sequences[0][0])
        expected = int
        assert  actual == expected


    def test_nextWord_tokens_are_int(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        actual = type(sp.indexed_nextWord[0])
        expected = int
        assert  actual == expected

    def test_warning_for_sequence_longer_than_text(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        expected = len(text.split(' ')) - 1
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        with pytest.warns(UserWarning):
            sp.text_to_sequences(100)

class TestTextToWordSequence():

    def test_next_word_is_last_word_in_previous_sequence(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_word_sequences(4)
        last_token_in_sequence = sp.word_sequences[1][-1]
        next_word = sp.word_nextWord[0]
        assert last_token_in_sequence == next_word

    def test_number_of_sequences_equals_number_nextWord(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_word_sequences(4)
        sequence_length = len(sp.word_sequences)
        nextword_length = len(sp.word_nextWord)
        assert  sequence_length == nextword_length


    def test_next_word_is_one_less_than_text_string(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        expected = len(text.split(' ')) - 1
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_word_sequences(4)
        actual = len(sp.word_nextWord)
        assert actual == expected


    def test_sequence_tokens_are_string(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_word_sequences(4)
        actual = type(sp.word_sequences[0][0])
        expected = str
        assert  actual == expected


    def test_nextWord_tokens_are_string(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        sp.text_to_word_sequences(4)
        actual = type(sp.word_nextWord[0][0])
        expected = str
        assert  actual == expected


    def test_warning_for_sequence_longer_than_text(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.tokenizeWords()
        with pytest.warns(UserWarning):
            sp.text_to_word_sequences(100)

class TestWordFrequencyLength():

    def test_correct_frequency(self):
        test_case = 'ctrl'
        text = 'This is is a a a sample sample text text string for for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.word_frequency_length()
        actual = sp.word_frequencies
        expected = [1,2,2,3,3,3,2,2,2,2,1,2,2,1,1]
        assert actual == expected

    def test_frequency_for_each_word(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.word_frequency_length()
        actual = len(sp.word_frequencies)
        expected = 9
        assert actual == expected


    def test_correct_word_length(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.word_frequency_length()
        actual = sp.word_lengths
        expected = [4,2,1,6,4,6,3,7,8]
        assert actual == expected


    def test_length_for_each_word(self):
        test_case = 'ctrl'
        text = 'This is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        sp.word_frequency_length()
        actual = len(sp.word_lengths)
        expected = 9
        assert actual == expected

class TestSemanticDissimlarity():

    def test_semantic_dissimilarity_returns_list(self):
        test_case = 'ctrl'
        text = 'this is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        embedding_filepath = 'Word_embeddings/glove.6B.50d.txt'
        sp.text_to_word_sequences(5)
        sp.calc_semantic_dissimilarity(embedding_filepath)
        actual = type(sp.Semantic_disimilarity)
        expected = list
        assert actual == expected


    def test_semantic_dissimilarity_equals_text_length(self):
        test_case = 'ctrl'
        text = 'this is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        embedding_filepath = 'Word_embeddings/glove.6B.50d.txt'
        sp.text_to_word_sequences(5)
        sp.calc_semantic_dissimilarity(embedding_filepath)
        actual = len(sp.Semantic_disimilarity)
        expected = 9
        assert actual == expected

    def test_semantic_dissimilarity_chech_no_value_nan(self):
        test_case = 'ctrl'
        text = 'this is a sample text string for testing purposes'
        sp = sequence_processor(test_case, text)
        embedding_filepath = 'Word_embeddings/glove.6B.50d.txt'
        sp.text_to_word_sequences(5)
        sp.calc_semantic_dissimilarity(embedding_filepath)
        data = pd.Series(sp.Semantic_disimilarity)
        actual = data.isna().sum()
        expected = 0
        assert actual == expected

