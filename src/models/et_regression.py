from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd

def linear_regression(x_train, y_train, x_test, y_test):
    """ performs linear regression and calculates the R squred score

    Parameters
    ----------
    x_train: array
        array containing the training features for the model

    y_train: array
        array containing the targets for training the model

    x_test: array
        array containing the testing features for the model

    y_test: array
        array containing the targets for testing the model

    Returns:
    -------
    R_squared: float
        coefficient of determination of the model
        """

    model = LinearRegression()
    model.fit(x_train,y_train)
    R_squared = model.score(x_test,y_test)
    return R_squared


def K_fold_validation(fold, x,y):
    """ performs k-fld cross validation and returns a list of R squared
        score for each fold.

    Parameters
    ----------
    fold: int
        the number of folds to cross validate the model on

    x: pandas.DataFrame
        the features of the data

    y: pandas.DataFrame
        the targets of the data

    Returns:
    -------
    R_scores: list
        list of the r squared score for each fold
    """
    kf = KFold(n_splits=fold)
    R_scores = []
    for i in range(y.shape[1]):
        Eye_measure_r_score = 0
        target = y.iloc[:, i]
        for train_index, test_index in kf.split(x):
            x_train, x_test = x.values[train_index], x.values[test_index]
            target_train, target_test = target.values[train_index], target.values[test_index]
            r_score = linear_regression(x_train, target_train, x_test, target_test)
            Eye_measure_r_score += r_score
        R_scores.append(np.true_divide(Eye_measure_r_score, fold))

    return R_scores


def compute_RSquared(y_true, predictions):
    """ compute the r squared score based on the predictions
    and the true values.

     Parameters
     ----------
     y_true: int
         the true targets of the data

     predictions: array
         the predictions of the model

     Returns:
     -------
     rsquared: float
         coefficient of determination
   """


    y_mean = np.mean(y_true, axis=0).reshape(1, y_true.shape[1])
    total_sum_squares = np.sum(np.square((y_true - y_mean)))
    residual_sum_squares = np.sum(np.square(predictions - y_true))
    rsquared = 1-(residual_sum_squares/total_sum_squares)
    return rsquared


def flatten_df(data_frame):
    """converts a multi column data frame to a two column dataframe where the
        first column in the data frame consists of the column names of the input
        dataframe. This allows ease of plotting boxplots.

    Parameters:
    ----------
    data_frame: pandas DataFrame
            mutlit colum dataframe.

    Returns:
    --------
    two_col_dataframe: pandas.Dataframe
        two column dataframe where the first column in the data frame consists of the column names of the input
        dataframe
    """

    two_col_dataframe = pd.DataFrame()
    for names in data_frame.columns:
        col = data_frame[names]
        cat_frame = pd.DataFrame(data={'Metric': names, 'R_squared': col})
        two_col_dataframe = pd.concat([two_col_dataframe, cat_frame])
    return two_col_dataframe