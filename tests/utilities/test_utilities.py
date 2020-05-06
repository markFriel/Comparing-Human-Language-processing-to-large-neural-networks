import pytest
import pandas as pd
import src.utilities.utilities_code as uc

class TestFlattenWordList():

    @pytest.fixture()
    def create_data(self, tmpdir):
        filepath = tmpdir.join("data")
        feature = [1,2,3,4]
        data_frame = pd.DataFrame(feature)
        data_frame.to_csv(filepath, index=False)
        yield filepath

    def test_return_expected_list(self,create_data):
        actual = uc.read_flaten_word_list(create_data)
        expected = [0,1,2,3,4]
        assert actual == expected

    def test_return_type(self,create_data):
        actual = type(uc.read_flaten_word_list(create_data))
        expected = list
        assert actual == expected

class TestReadDirectory():

    def test_return_length(self):
        actual = len(uc.read_directory('Results'))
        expected = 2
        assert actual == expected

    def test_does_not_contain_DS_STore(self):
        actual = uc.read_directory('Results')
        expected = ['EEG_Data','Reading_Data']
        assert actual == expected

    def test_does_not_contain_ipynb(self):
        actual = uc.read_directory('Results')
        expected = ['EEG_Data', 'Reading_Data']
        assert actual == expected


