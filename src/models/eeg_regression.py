from mne.evoked import EvokedArray
from sklearn.linear_model import  Ridge
from sklearn.model_selection import KFold
from scipy import sparse
import numpy as np
from mne.io.pick import _picks_to_idx, pick_info
from mne.utils import _reject_data_segments


class TimeResolvedRegression:


    def __init__(self,eeg_data, events,  event_id,covariates,tmin, tmax ,picks=None, decim=1):
        self.raw, self.info, self.events = self.prepare_rerp_data(eeg_data, events, picks=picks, decim=decim)
        predictor_matrix, self.conds, self.cond_length, self.tmin_s, self.tmax_s = \
                                                                    self.prepare_rerp_preds(n_samples=self.raw.shape[1],
                                                                    sfreq=128, events = self.events, event_id = event_id,
                                                                    tmin = tmin, tmax = tmax, covariates = covariates)
        self.design_matrix, self.expected_data = self.clean_rerp_input(predictor_matrix, self.raw, reject=None, flat=None, decim=1,info=self.info, tstep=1)

    def prepare_rerp_data(self,raw, events, picks=None, decim=1):
        """Prepare events and data, primarily for `linear_regression_raw`."""
        picks = _picks_to_idx(raw.info, picks)
        info = pick_info(raw.info, picks)
        decim = int(decim)
        info["sfreq"] /= decim
        data, times = raw[:]
        data = data[picks, ::decim]
        if len(set(events[:, 0])) < len(events[:, 0]):
            raise ValueError("`events` contains duplicate time points. Make "
                             "sure all entries in the first column of `events` "
                             "are unique.")

        events = events.copy()
        events[:, 0] -= raw.first_samp
        events[:, 0] //= decim
        if len(set(events[:, 0])) < len(events[:, 0]):
            raise ValueError("After decimating, `events` contains duplicate time "
                             "points. This means some events are too closely "
                             "spaced for the requested decimation factor. Choose "
                             "different events, drop close events, or choose a "
                             "different decimation factor.")

        return data, info, events


    def prepare_rerp_preds(self,n_samples, sfreq, events, event_id=None, tmin=-.1,tmax=1, covariates=None):
        """Build predictor matrix and metadata (e.g. condition time windows)."""
        conds = list(event_id)
        if covariates is not None:
            conds += list(covariates)

        # time windows (per event type) are converted to sample points from times
        # int(round()) to be safe and match Epochs constructor behavior
        if isinstance(tmin, (float, int)):
            tmin_s = {cond: int(round(tmin * sfreq)) for cond in conds}
        else:
            tmin_s = {cond: int(round(tmin.get(cond, -.1) * sfreq))
                      for cond in conds}
        if isinstance(tmax, (float, int)):
            tmax_s = {
                cond: int(round((tmax * sfreq)) + 1) for cond in conds}
        else:
            tmax_s = {cond: int(round(tmax.get(cond, 1.) * sfreq)) + 1
                      for cond in conds}

        # Construct predictor matrix
        # We do this by creating one array per event type, shape (lags, samples)
        # (where lags depends on tmin/tmax and can be different for different
        # event types). Columns correspond to predictors, predictors correspond to
        # time lags. Thus, each array is mostly sparse, with one diagonal of 1s
        # per event (for binary predictors).

        cond_length = dict()
        xs = []
        for cond in conds:
            tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
            n_lags = int(tmax_ - tmin_)  # width of matrix
            if cond in event_id:  # for binary predictors
                ids = ([event_id[cond]]
                       if isinstance(event_id[cond], int)
                       else event_id[cond])
                onsets = -(events[np.in1d(events[:, 2], ids), 0] + tmin_)
                values = np.ones((len(onsets), n_lags))

            else:  # for predictors from covariates, e.g. continuous ones
                covs = covariates[cond]
                if len(covs) != len(events):
                    error = ("Condition {0} from ```covariates``` is "
                             "not the same length as ```events```").format(cond)
                    raise ValueError(error)
                onsets = -(events[np.where(covs != 0), 0] + tmin_)[0]
                v = np.asarray(covs)[np.nonzero(covs.values)].astype(float)
                values = np.ones((len(onsets), n_lags)) * v[:, np.newaxis]

            cond_length[cond] = len(onsets)
            xs.append(sparse.dia_matrix((values, onsets),shape=(n_samples, n_lags)))
        return sparse.hstack(xs), conds, cond_length, tmin_s, tmax_s


    def clean_rerp_input(self,X, data, reject, flat, decim, info, tstep):
        """Remove empty and contaminated points from data & predictor matrices."""
        # find only those positions where at least one predictor isn't 0
        has_val = np.unique(X.nonzero()[0])

        # reject positions based on extreme steps in the data
        if reject is not None:
            _, inds = _reject_data_segments(data, reject, flat, decim=None,
                                            info=info, tstep=tstep)
            for t0, t1 in inds:
                has_val = np.setdiff1d(has_val, range(t0, t1))

        return X.tocsr()[has_val], data[:, has_val]


def _make_evokeds(coefs, conds, cond_length, tmin_s, tmax_s, info):
    """Create a dictionary of Evoked objects.

    These will be created from a coefs matrix and condition durations.
    """
    evokeds = dict()
    cumul = 0
    for cond in conds:
        tmin_, tmax_ = tmin_s[cond], tmax_s[cond]
        evokeds[cond] = EvokedArray(
            coefs[:, cumul:cumul + tmax_ - tmin_], info=info, comment=cond,
            tmin=tmin_ / float(info["sfreq"]), nave=cond_length[cond],
            kind='average')  # nave and kind are technically incorrect
        cumul += tmax_ - tmin_
    return evokeds


def grand_average_coef(list_of_arrays):
    """Sums a list of numpy arrays and finds the average
    Parameters:
    ----------
    list_of_arrays: list
        a list containg arrays, values are the coefficients of the model

    Returns:
    --------
    average_coefficients: numpy array
        array of the average values form the list of arrays

    """

    summed_values = np.zeros((list_of_arrays[0].shape))
    for i in range(len(list_of_arrays)):

        summed_values += list_of_arrays[i]

    average_coefficients =summed_values / len(list_of_arrays)

    return average_coefficients


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


def KFold_cv(fold, x, y):
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

    metrics =[]
    kfold = KFold(n_splits=fold)
    for train_index, test_index in kfold.split(x):
        x_train, y_train, x_test, y_test = x.iloc[train_index], y.iloc[train_index], x.iloc[test_index], y.iloc[test_index]
        rmodel = Ridge(alpha=1000, fit_intercept=False).fit(x_train, y_train)
        r_score = rmodel.score(x_test, y_test)
        metrics.append(r_score)
    return metrics


def save_ERPS(Evoked_dictionary, folderpath):
    """ Saves the elements of a dictionary to a file
    Parameters:
    -----------
    Evoked_dictionary: dictionary
        A dictionary of evoked array objects

    folderpath : str
        path to save the evoked arrays
    """


    for i in Evoked_dictionary.keys():
        ERP = Evoked_dictionary[i]
        path = folderpath.join(i) +('ERP-ave.fif')
        ERP.save(path)




