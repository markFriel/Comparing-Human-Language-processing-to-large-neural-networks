import pytest
import src.models.et_regression as linreg
import numpy as np
import pandas as pd


class TestLinearRegression:

    def test_on_linear_data(self):
        train_argument = np.array([[1.0, 3.0], [2.0, 5.0], [3.0, 7.0]])
        train_target = np.array([[3.0], [5.0], [7.0]])

        test_argument = np.array([[4.0, 9.0], [5.0, 11.0], [6.0, 13.0]])
        test_target = np.array([[9.0], [11.0], [13.0]])

        actual = linreg.linear_regression(train_argument, train_target, test_argument, test_target)
        expected = 1.0
        assert actual == pytest.approx(expected)

    def test_on_non_linear_data(self):
        train_argument = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        train_target = np.array([[1.0], [1.0], [1.0]])

        test_argument = np.array([[4.0, 2.0], [5.0, 2.0], [6.0, 2.0]])
        test_target = np.array([[2.0], [2.0], [2.0]])

        actual = linreg.linear_regression(train_argument, train_target, test_argument, test_target)
        expected = 0.0
        assert actual == pytest.approx(expected)

    def test_return_type(self):
        train_argument = np.array([[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]])
        train_target = np.array([[1.0], [1.0], [1.0]])

        test_argument = np.array([[4.0, 2.0], [5.0, 2.0], [6.0, 2.0]])
        test_target = np.array([[2.0], [2.0], [2.0]])

        actual = type(linreg.linear_regression(train_argument, train_target, test_argument, test_target))
        expected = float
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

    @pytest.fixture
    def data_setup_two_column_y(self):
        x_data = pd.DataFrame(np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0], [5.0, 5.0], [6.0, 6.0]]))
        y_data = pd.DataFrame([[2.0,2.0], [4.0,4.0], [6.0,6.0], [8.0,8.0], [10.0,10.0], [12.0,12.0]])
        yield x_data, y_data

    def test_value_of_one_column(self, data_setup_one_column):
        x_data, y_data = data_setup_one_column
        fold = 2
        actual = (linreg.K_fold_validation(fold, x_data, y_data))[0]
        expeceted = 1.0
        assert actual == pytest.approx(expeceted)

    def test_value_return_type(self, data_setup_two_column):
        x_data, y_data = data_setup_two_column
        fold = 2
        actual = type(linreg.K_fold_validation(fold, x_data, y_data))
        expected = list
        assert actual == expected

    def test_value_of_two_column(self, data_setup_two_column_y):
        x_data, y_data = data_setup_two_column_y
        fold = 2
        actual = len((linreg.K_fold_validation(fold, x_data, y_data)))
        expected = 2
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
        actual = linreg.compute_RSquared(y_true, predictions)
        expected = 0.0
        assert actual == pytest.approx(expected)

    def test_r_squared_value_true_values(self, setup_true_predictions):
        predictions, y_true = setup_true_predictions
        actual = linreg.compute_RSquared(y_true, predictions)
        expected = 1.0
        assert actual == pytest.approx(expected)

    def test_r_squared_value_type(self, setup_false_predictions):
        predictions, y_true = setup_false_predictions
        actual = type(linreg.compute_RSquared(y_true, predictions))
        expected = np.float64
        assert actual == expected

class TestFalttenDF():
    @pytest.fixture
    def set_up_data(self):
        feature1 = [[1.1], [1.2], [1.3], [1.4]]
        feature2 = [[2.1], [2.2], [2.3], [2.4]]
        feature3 = [[3.1], [3.2], [3.3], [3.4]]
        feature4 = [[4.1], [4.2], [4.3], [4.4]]
        data_frame= pd.DataFrame(data={'model1':feature1, 'model2':feature2, 'model3':feature3, 'model4':feature4})
        yield data_frame

    def test_dataframe_dimensions(self,set_up_data):
        actual = linreg.flatten_df(set_up_data).shape[1]
        expected = 2
        assert actual == expected

    def test_dataframe_return_type(self,set_up_data):
        actual = type(linreg.flatten_df(set_up_data))
        expected = pd.DataFrame
        assert actual == expected

