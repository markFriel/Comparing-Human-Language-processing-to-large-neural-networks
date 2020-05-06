import pytest
import src.models.eeg_regression as er
import mne
import os
import pandas as pd
from os import path
import numpy as np


class TestSaveERPS:

    @pytest.fixture
    def set_up_data(self):
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
        sampling_rate = 100
        channel_types = 'eeg'
        info = mne.create_info(channel_names, sampling_rate, channel_types)
        np.random.seed(1)
        data = np.random.randn(10, 128, sampling_rate * 2)

        data_evoked = data.mean(0)
        tmin = -0.1
        nave = data.shape[0]

        comment = "Smiley faces"

        # Create the Evoked object
        evoked_array = mne.EvokedArray(data_evoked, info, tmin, comment=comment, nave=nave)
        evoked_array.set_montage('biosemi128')
        evoked_dictionary = {'Evoked_Array_1': evoked_array}
        yield evoked_dictionary

    def test_save_files(self, set_up_data, tmpdir):
        er.save_ERPS(set_up_data, tmpdir)
        file_path = tmpdir.join('Evoked_Array_1ERP-ave.fif')
        print(file_path)
        actual = path.exists(file_path)
        print(os.listdir(tmpdir))
        expected = True
        assert actual == expected

    def test_false_file_path(self, set_up_data, tmpdir):
        er.save_ERPS(set_up_data, tmpdir)
        file_path = tmpdir.join('false')
        actual = path.exists(file_path)
        expected = False
        assert actual == expected

class TestGrandAveragedCoefs:

    @pytest.fixture
    def data_set_up(self):
        list_of_arrays = []
        for i in range(10):
            list_of_arrays.append(np.random.randint(1,4,(128,310)))

        yield list_of_arrays

    def test_return_type(self, data_set_up):
       actual = type(er.grand_average_coef(data_set_up))
       expected = np.ndarray
       assert actual == expected

    def test_element_return_type(self, data_set_up):
        actual = type(er.grand_average_coef(data_set_up)[0][0])
        expected = np.float64
        assert actual == expected

class TestComputeRsquared:

    @pytest.fixture
    def setup_true_predictions(self):
        predictions = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
        yield predictions, y_true

    @pytest.fixture
    def setup_false_predictions(self):
        predictions = np.array([[4.0], [4.0], [2.0], [4.0], [4.0], [4.0]])
        y_true = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [9.0]])
        yield predictions, y_true

    def test_r_squared_value(self, setup_false_predictions):
        predictions, y_true = setup_false_predictions
        actual = er.compute_RSquared(y_true, predictions)
        expected = 0.0
        assert actual == pytest.approx(expected)

    def test_r_squared_value_true_values(self, setup_true_predictions):
        predictions, y_true = setup_true_predictions
        actual = er.compute_RSquared(y_true, predictions)
        expected = 1.0
        assert actual == pytest.approx(expected)

    def test_r_squared_value_type(self, setup_false_predictions):
        predictions, y_true = setup_false_predictions
        actual = type(er.compute_RSquared(y_true, predictions))
        expected = np.float64
        assert actual == expected

class TestKFoldValidation:

    @pytest.fixture
    def data_setup_one_column(self):
        x_data = pd.DataFrame(np.array([[1.0], [3.0], [2.0], [5.0], [3.0], [7.0], [8.0], [9.0], [10.0]]))
        y_data = pd.DataFrame(np.array([[2.0], [6.0], [4.0], [10.0], [6.0], [14.0], [16.0], [18.0], [20.0]]))
        yield x_data, y_data

    @pytest.fixture
    def data_setup_two_column(self):
        x_data = pd.DataFrame(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]))
        y_data = pd.DataFrame([[2.0], [4.0], [6.0], [8.0], [10.0], [12.0]])
        yield x_data, y_data

    @pytest.mark.xfail
    def test_value_of_one_column(self, data_setup_one_column):
        x_data, y_data = data_setup_one_column
        fold = 2
        actual = (er.KFold_cv(fold, x_data, y_data))[0]
        expeceted = 1.0
        assert actual == pytest.approx(expeceted)

    def test_value_return_type(self, data_setup_two_column):
        x_data, y_data = data_setup_two_column
        fold = 2
        actual = type(er.KFold_cv(fold, x_data, y_data))
        expected = list
        assert actual == expected

    def test_value_of_two_column(self, data_setup_two_column):
        x_data, y_data = data_setup_two_column
        fold = 2
        actual = len((er.KFold_cv(fold, x_data, y_data)))
        expected = 2
        assert actual == expected

