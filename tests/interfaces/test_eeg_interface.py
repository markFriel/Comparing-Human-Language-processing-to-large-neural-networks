import pytest
import src.interfaces.eeg_interface as eei
from random_words import RandomWords
import pandas as pd
import numpy as np
import scipy
from scipy.io import savemat
import mne
import os


class TestCalculateCovariates:

    @pytest.fixture
    def set_up_data(self):
        rw = RandomWords()
        randomwords = rw.random_words(count=50)
        yield randomwords

    def test_length_surprisal_gpt2(self, set_up_data):
        randomwords = set_up_data
        surprisal = eei.calculate_covariates('gpt2', randomwords, 5)
        actual = len(surprisal)
        expected = len(set_up_data)
        assert actual == expected

    def test_value_type_of_surprisal(self, set_up_data):
        randomwords = set_up_data
        surprisal = eei.calculate_covariates('gpt2', randomwords, 5)
        actual = type(surprisal)
        expected = pd.DataFrame
        assert actual == expected

    def test_range_values_of_surprisal(self, set_up_data):
        randomwords = set_up_data
        surprisal = eei.calculate_covariates('gpt2', randomwords, 5)
        surprisal_list = surprisal['Surprisal'].values.tolist()
        if all(x >= 0 for x in surprisal_list):
            actual = True
        else:
            actual = False

        expected = True
        assert actual == expected


class TestPreprocessedEEG:

    @pytest.fixture
    def set_up_data(self, tmpdir):
        data = np.random.randint(1,1000,(1000,128))
        mastoids = np.random.randint(1,1000,(1000,2))
        channel_name_filepath = tmpdir.join('channel_names.txt')
        with open(channel_name_filepath, "w") as f:
            f.write(
                "A1\n""A2\n""A3\n""A4\n""A5\n""A6\n""A7\n""A8\n""A9\n""A10\n""A11\n""A12\n""A13\n""A14\n""A15\n""A16\n"
                "A17\n""A18\n""A19\n""A20\n""A21\n""A22\n""A23\n""A24\n""A25\n""A26\n""A27\n""A28\n""A29\n""A30\n""A31\n""A32\n"
                "B1\n""B2\n""B3\n""B4\n""B5\n""B6\n""B7\n""B8\n""B9\n""B10\n""B11\n""B12\n""B13\n""B14\n""B15\n""B16\n"
                "B17\n""B18\n""B19\n""B20\n""B21\n""B22\n""B23\n""B24\n""B25\n""B26\n""B27\n""B28\n""B29\n""B30\n""B31\n""B32\n"
                "C1\n""C2\n""C3\n""C4\n""C5\n""C6\n""C7\n""C8\n""C9\n""C10\n""C11\n""C12\n""C13\n""C14\n""C15\n""C16\n"
                "C17\n""C18\n""C19\n""C20\n""C21\n""C22\n""C23\n""C24\n""C25\n""C26\n""C27\n""C28\n""C29\n""C30\n""C31\n""C32\n"
                "D1\n""D2\n""D3\n""D4\n""D5\n""D6\n""D7\n""D8\n""D9\n""D10\n""D11\n""D12\n""D13\n""D14\n""D15\n""D16\n"
                "D17\n""D18\n""D19\n""D20\n""D21\n""D22\n""D23\n""D24\n""D25\n""D26\n""D27\n""D28\n""D29\n""D30\n""D31\n""D32\n"
            )

        filepath = 'Output/eeg_file.mat'

        dictionary_variables = {'eegData':data, 'mastoids':mastoids}
        savemat(filepath, dictionary_variables)

        event_filepath = 'Output/event_file.mat'
        events = np.array([[0],[100],[200],[300],[400],[500],[600],[700],[800]])
        event_dictionary = {'onset_time':events}
        savemat(event_filepath, event_dictionary)


        yield filepath, channel_name_filepath, event_filepath
        os.remove(filepath)
        os.remove(event_filepath)


    def test_preprocess_return_type(self, set_up_data):
        matlab_file , channel_file, event_path = set_up_data
        raw, events = eei.preprocessed_eeg_raw(channel_file, matlab_file, 'biosemi128', event_path, 'onset_time')
        actual_raw = type(raw)
        expected_raw = mne.io.RawArray

        actual_events = type(events)
        expected_events = np.ndarray
        assert actual_raw == expected_raw
        assert actual_events == expected_events

    def test_preprocess_data_shape(self, set_up_data):
        matlab_file, channel_file, event_path = set_up_data
        raw, events = eei.preprocessed_eeg_raw(channel_file, matlab_file, 'biosemi128', event_path, 'onset_time')
        actual_raw = raw._data.shape
        expected_raw = (128,1000)

        actual_events = events.shape
        expected_events = (9,3)
        print(events)
        assert actual_raw == expected_raw
        assert actual_events == expected_events

class TestTimeResolvedRegression:

    @pytest.fixture()
    def setup_data(self):
        data = np.random.randint(1,1000,(128,1000))
        channel_names = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "A14", "A15",
                         "A16",
                         "A17", "A18", "A19", "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29",
                         "A30", "A31", "A32",
                         "B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12", "B13", "B14", "B15",
                         "B16",
                         "B17", "B18", "B19", "B20", "B21", "B22", "B23", "B24", "B25", "B26", "B27", "B28", "B29",
                         "B30", "B31", "B32",
                         "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15",
                         "C16",
                         "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26", "C27", "C28", "C29",
                         "C30", "C31", "C32",
                         "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
                         "D16",
                         "D17", "D18", "D19", "D20", "D21", "D22", "D23", "D24", "D25", "D26", "D27", "D28", "D29",
                         "D30", "D31", "D32"]
        channel_type = 'eeg'
        sampling_freq =100
        info = mne.create_info(channel_names, ch_types=channel_type, sfreq=sampling_freq)
        raw = mne.io.RawArray(data, info)

        events = np.array([[0, 0, 1],[100, 0, 1],
                            [200, 0, 1],[300, 0, 1],
                            [400, 0, 1],[500, 0, 1],
                            [600, 0, 1],[700, 0, 1],
                            [800, 0, 1],[900, 0, 1],])

        covariates = pd.DataFrame(data={'Surprisal': [1,2,3,4,5,6,7,8,9,10]})
        event_id = {'onset_time': 1}

        yield raw, events, event_id, covariates

    def test_return_type(self, setup_data):
        raw, events, event_id, covariates = setup_data
        coefficients , regression_data = eei.time_resolved_regression(raw, events, event_id, covariates)
        actual_coef = type(coefficients)
        expected_coef = np.ndarray

        actual_reg_data = type(regression_data.design_matrix)
        expected_reg_data = scipy.sparse.csr.csr_matrix
        assert actual_coef == expected_coef
        assert actual_reg_data == expected_reg_data


    def test_return_shape(self,setup_data):
        raw, events, event_id, covariates = setup_data
        coefficients, regression_data = eei.time_resolved_regression(raw, events, event_id, covariates)
        actual_coef = coefficients.shape
        expected_coef = (128,310)

        actual_reg_data = regression_data.design_matrix.shape
        expected_reg_data = (1000,310)
        assert actual_coef == expected_coef
        assert actual_reg_data == expected_reg_data




