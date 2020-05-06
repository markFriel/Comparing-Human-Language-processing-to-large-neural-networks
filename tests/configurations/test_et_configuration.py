import pytest
import pandas as pd
from random_words import RandomWords
import src.configurations.ET_Run_configuration as etc
import os
from os import path

class TestAggregatedSubjectConfiguration:


    @pytest.fixture
    def data_set_up(self):
        test_dataframe = pd._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=10)
        words =[]
        for i in range(3):
            words.extend(rndwords)

        list = ['Sub_1'] * 10
        sub_2 = ['Sub_2'] * 10
        sub_3 = ['Sub_3'] * 10

        list.extend(sub_2)
        list.extend(sub_3)

        test_dataframe['A'] = list
        test_dataframe['B'] = words
        yield 'gpt2', test_dataframe, ['Sub_1','Sub_2','Sub_3'], 'A','B', 'C', 'Output/'

    def test_presence_file(self, data_set_up):
        model_type, reading_data, subject_IDs, subject_column, text_column, metric_column, output_file  = data_set_up
        etc.aggregated_data_analysis(model_type, 10, reading_data, subject_IDs, subject_column,
                                     text_column, metric_column, -3, output_file,)

        table_path = output_file + model_type + '.csv'
        actual = path.exists(table_path)
        expected = True
        assert actual == expected
        os.remove(table_path)

class TestIndividualSubjects:

    @pytest.fixture
    def data_set_up(self):
        test_dataframe = pd._testing.makeDataFrame()
        rw = RandomWords()
        rndwords = rw.random_words(count=10)
        words = []
        for i in range(3):
            words.extend(rndwords)

        list = ['Sub_1'] * 10
        sub_2 = ['Sub_2'] * 10
        sub_3 = ['Sub_3'] * 10

        list.extend(sub_2)
        list.extend(sub_3)

        test_dataframe['A'] = list
        test_dataframe['B'] = words
        yield 'gpt2', test_dataframe, ['Sub_1', 'Sub_2', 'Sub_3'], 'A', 'B', 'C', 'Output/'

    def test_graph_individual_subjects(self,data_set_up ):
        model_type, reading_data, subject_IDs, subject_column, text_column, metric_column, output_file = data_set_up
        etc.individual_data_analysis(model_type, 10,  reading_data, subject_IDs, subject_column,
                                     text_column, metric_column, -3, output_file)

        plot_path = output_file + 'C.png'
        actual_c = path.exists(plot_path)
        expected_c = True
        assert actual_c == expected_c
        os.remove(plot_path)


        plot_path = output_file + 'D.png'
        actual_d = path.exists(plot_path)
        assert actual_d == True
        os.remove(plot_path)


