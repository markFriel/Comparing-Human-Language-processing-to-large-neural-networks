from scipy.stats import spearmanr
from scipy.stats import shapiro

def normality_test(data_frame):
    """ Test the normal distribution of data using Shapiro-wilk test

    Parameters
    ----------
    data_frame : pandas dataframe
        A pandas dataframe of one or more columns

    Outputs
    -------
        For each column it will output whether the data is normally distributed,
        whether to reject the null hypothesis and the p-value.
    """

    columns = data_frame.columns
    for i in range(len(columns)):
        values = data_frame[columns[i]].values.tolist()
        # Shapiro-Wilk Test normality test
        stat, p = shapiro(values)
        #print(columns[i], '\nStatistics=%.3f, p=%.3f' % (stat, p))
        alpha = 0.05
        if p > alpha:
            print('Sample looks Gaussian (fail to reject H0)\n')
        else:
            print('Sample does not look Gaussian (reject H0)\n')


def calculate_Spearmans(data_frame, features1 , features2):
    """ Calcultes the correlation between each feature in feature1 to every feature in feature 2

        Parameters
        ----------
        data_frame : pandas dataframe
            A pandas dataframe of one or more columns

        features1 : list
            A list of strings that represen tot column names to be correlated
            to the columns in features2

        features2 : list
            A list of strings that represen tot column names to be correlated
            to the columns in features1

        Outputs
        -------
            The correlation between the data of twor features
        """
    for i in features1:
        var1 = data_frame[i].values.tolist()
        #print('\n' + i, 'Correlation\n')
        for j in features2:
            var2 = data_frame[j].values.tolist()
            corr = spearmanr(var1, var2)
            print(i.lower() + ' vs '+ j.lower() + ' s_corr: ' + str(corr[0]))


























