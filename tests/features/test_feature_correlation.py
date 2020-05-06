import pytest
import numpy as np
from scipy.stats import spearmanr
from scipy.stats import skewnorm
import pandas as pd
import src.features.feature_correlation as fc


class TestNormalityTest():

    @pytest.fixture
    def normalyDistributedData(self):
        normal_data = np.random.normal(50.0, 15, (1,100)).tolist()[0]
        data = pd.DataFrame(data={'Data':normal_data})
        yield data

    @pytest.fixture
    def NonNormalDistributedData(self):
        pers = np.arange(1, 101, 1)
        prob = [1.0] * (len(pers) - 41) + [5.0] * 41
        # Normalising to 1.0
        prob /= np.sum(prob)
        distribution = np.random.choice(pers, 100, p=prob)
        data = pd.DataFrame(data={'Data': distribution})
        yield data

    def test_on_normal_distribute_data(self,capsys,normalyDistributedData):
        fc.normality_test(normalyDistributedData)
        captured = capsys.readouterr()
        assert captured.out == 'Sample looks Gaussian (fail to reject H0)\n\n'

    def test_on_non_normal_data(self, capsys, NonNormalDistributedData):
        fc.normality_test(NonNormalDistributedData)
        captured = capsys.readouterr()
        assert captured.out == 'Sample does not look Gaussian (reject H0)\n\n'

    def test_input_checking(self):
        with pytest.raises(TypeError):
            fc.normality_test()


class TestCalculateSpearmans():

    @pytest.fixture
    def normalyDistributedData(self):
        normal_data = np.random.normal(50.0, 15, (1, 100)).tolist()[0]
        data = pd.DataFrame(data={'Data': normal_data})
        yield data

    @pytest.fixture
    def NonNormalDistributedData(self):
        pers = np.arange(1, 101, 1)
        prob = [1.0] * (len(pers) - 41) + [5.0] * 41
        # Normalising to 1.0
        prob /= np.sum(prob)
        distribution = np.random.choice(pers, 100, p=prob)
        data = pd.DataFrame(data={'Data': distribution})
        yield data

    @pytest.fixture
    def Correlated_feature_setup(self):
        feature1 = [1,2,3,4,5,6]
        feature2 = [200,300,400,500,600,700]
        data = pd.DataFrame(data={'feature1': feature1, 'feature2':feature2})
        yield data

    @pytest.fixture
    def Non_Correlated_feature_setup(self):
        feature1 = [1, 2, 3, 4, 5, 6]
        feature2 = [1, 2, 1, 1, 2, 1]
        spearmanr(feature1,feature2)
        data = pd.DataFrame(data={'feature1': feature1, 'feature2': feature2})
        yield data

    def test_input_checking(self):
        with pytest.raises(TypeError):
            fc.calculate_Spearmans()


    def test_execution_spearmans_correlation(self,capsys,Correlated_feature_setup):
        fc.calculate_Spearmans(Correlated_feature_setup, ['feature1'],['feature2'])
        captured = capsys.readouterr()
        assert captured.out == 'feature1 vs feature2 ' + 's_corr: 1.0\n'


    def test_execution_spearmans_correlation_non_correlated(self,capsys,Non_Correlated_feature_setup):
        fc.calculate_Spearmans(Non_Correlated_feature_setup, ['feature1'],['feature2'])
        captured = capsys.readouterr()
        assert captured.out == 'feature1 vs feature2 ' + 's_corr: 0.0\n'





