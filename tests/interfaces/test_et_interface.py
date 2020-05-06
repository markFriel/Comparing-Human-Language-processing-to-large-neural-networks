import pytest
import pandas as pd
import src.interfaces.et_interface as eti
from random_words import RandomWords


class TestCalculateCovariates:

    @pytest.fixture
    def set_up_data(self):
        rw = RandomWords()
        randomwords = rw.random_words(count=50)
        yield randomwords

    def test_length_surprisal_gpt2(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = len(surprisal)
        expected = len(set_up_data)
        assert actual == expected

    def test_length_word_length_gpt2(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = len(word_length)
        expected = len(set_up_data)
        assert actual == expected

    def test_length_word_frequency_gpt2(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = len(word_frequency)
        expected = len(set_up_data)
        assert actual == expected

    def test_value_type_of_surprisal(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = type(surprisal[0])
        expected = float
        assert actual == expected

    def test_value_type_of_length(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = type(word_length[0])
        expected = int
        assert actual == expected

    def test_value_type_of_frequency(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        actual = type(word_frequency[0])
        expected = int
        assert actual == expected

    def test_range_values_of_surprisal(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        if all(x >= 0 for x in surprisal):
            actual = True
        else:
            actual = False

        expected = True
        assert actual == expected

    def test_range_values_of_length(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        if all(x >= 1 for x in word_length):
            actual = True
        else:
            actual = False

        expected = True
        assert actual == expected

    def test_range_values_of_frequency(self, set_up_data):
        randomwords = set_up_data
        surprisal, word_length, word_frequency = eti.calculate_covariates('gpt2', randomwords, 5)
        if all(x >= 1 for x in word_frequency):
            actual = True
        else:
            actual = False

        expected = True
        assert actual == expected


class TestPreprocessingReadingData:

    @pytest.fixture()
    def data_set_up(self):
        test_dataframe = pd._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=30)
        list = []
        id = 1
        for i in range(30):
            if id > 4:
                id = 1
            subject_id = 'Sub_' + str(id)
            list.append(subject_id)
            id += 1

        test_dataframe['A'] = list
        test_dataframe['B'] = rndwords
        yield test_dataframe

    def test_return_type(self, data_set_up):
        eye_measure, word_list = eti.preprocess_reading_data(data_set_up, 'B', 'C', 1)
        actual_eye = type(eye_measure)
        expected_eye = pd.DataFrame

        actual_word = type(word_list)
        expected_word = list

        assert actual_eye == expected_eye
        assert actual_word == expected_word

    def test_element_return_type_word(self, data_set_up):
        eye_measure, word_list = eti.preprocess_reading_data(data_set_up, 'B', 'C', 1)
        print(word_list)
        actual = type(word_list[0])
        expected = str
        assert actual == expected




