import pytest
from  src.data.et_preprocessing import eye_tracking_preprocessing
import pandas
from random_words import RandomWords
import numpy as np


class TestSeperateSubjects():

    @pytest.fixture
    def reading_data_setup(self):
        test_dataframe = pandas._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=30)
        list =[]
        id = 1
        for i in range(30):
            if(id > 4):
                id =1
            subject_id = 'Sub_'+ str(id)
            list.append(subject_id)
            id+=1

        test_dataframe['A'] = list
        test_dataframe['B'] = rndwords
        yield test_dataframe


    def test_seperate_subject_data_splits(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids =['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        list_dataframes = dp._seperate_subject_data(reading_data_setup,ids, 'A')
        actual = len(list_dataframes)
        expected = 4
        assert actual == expected

    def test_seperate_subject_data_returns_list(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        list_dataframes = dp._seperate_subject_data(reading_data_setup, ids, 'A')
        actual = type(list_dataframes)
        expected = list
        assert actual == expected

    def test_seperate_subject_data_element_type(self,reading_data_setup ):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        list_dataframes = dp._seperate_subject_data(reading_data_setup, ids, 'A')
        actual = type(list_dataframes[0])
        expected = pandas.core.frame.DataFrame
        assert actual == expected

    def test_seperate_subject_data_df_length(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        list_dataframes = dp._seperate_subject_data(reading_data_setup, ids, 'A')
        actual = len(list_dataframes[0])
        expected = 8
        assert actual == expected

class TestAggregateSubjectData():

    @pytest.fixture
    def reading_data_setup(self):
        test_dataframe = pandas._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=30)
        list = []
        id = 1
        for i in range(30):
            if (id > 4):
                id = 1
            subject_id = 'Sub_' + str(id)
            list.append(subject_id)
            id += 1

        test_dataframe['A'] = list
        test_dataframe['B'] = rndwords
        yield test_dataframe

    def test_aggregate_subject_data_type(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        data_frame = dp.aggregate_subject_data(reading_data_setup, ids, 'A', 'B')
        actual = type(data_frame)
        expected = pandas.core.frame.DataFrame
        assert actual == expected

    def test_aggregate_subject_data_length(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        data_frame = dp.aggregate_subject_data(reading_data_setup, ids, 'A', 'B')
        actual = len(data_frame)
        expected = 8
        assert actual == expected

    def test_aggregate_subject_data_number_columns(self, reading_data_setup):
        dp = eye_tracking_preprocessing()
        ids = ['Sub_1', 'Sub_2', 'Sub_3', 'Sub_4']
        data_frame = dp.aggregate_subject_data(reading_data_setup, ids, 'A', 'B')
        actual = len(data_frame.columns)
        expected = 3
        assert actual == expected


class TestCorrectNullData():

    @pytest.fixture
    def reading_data_setup(self):
        test_dataframe = pandas._testing.makeMissingDataframe(density=0.6)
        rw = RandomWords()
        rndwords = rw.random_words(count=30)
        list = []
        id = 1
        for i in range(30):
            if (id > 4):
                id = 1
            subject_id = 'Sub_' + str(id)
            list.append(subject_id)
            id += 1

        test_dataframe['A'] = list
        test_dataframe['B'] = rndwords
        yield test_dataframe

    def test_no_null_data(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        output = dp.correct_null_data(reading_data_setup)
        actual = sum(output.isnull().sum(axis = 0))
        expected = 0
        assert actual == expected

    def test_no_null_data_type(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        data_frame = dp.correct_null_data(reading_data_setup)
        actual = type(data_frame)
        expected = pandas.core.frame.DataFrame
        assert actual == expected

    def test_no_null_data_length(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        data_frame = dp.correct_null_data(reading_data_setup)
        actual = len(data_frame)
        expected = 30
        assert actual == expected

class TestRemoveLexicalProcessing():

    @pytest.fixture
    def reading_data_setup(self):
        test_dataframe = pandas._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=30)
        list = []
        id = 1
        for i in range(30):
            if (id > 4):
                id = 1
            subject_id = 'Sub_' + str(id)
            list.append(subject_id)
            id += 1

        test_dataframe['A'] = list
        test_dataframe['B'] = rndwords
        yield test_dataframe

    def test_remove_lexical_type(self, reading_data_setup):
        dp = eye_tracking_preprocessing()
        data_frame = dp.remove_lexical_processing(reading_data_setup,'C', 1)
        actual = type(data_frame)
        expected = pandas.core.frame.DataFrame
        assert actual == expected

    def test_remove_lexical_length(self,reading_data_setup):
        dp = eye_tracking_preprocessing()
        data_frame = dp.remove_lexical_processing(reading_data_setup,'C', 1)
        actual = len(np.unique(np.asarray(data_frame['C'] < 1)))
        expected = 1
        assert actual == expected












