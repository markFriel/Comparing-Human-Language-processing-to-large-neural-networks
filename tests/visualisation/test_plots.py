import pytest
import pandas as pd
import src.visualisation.visualisation as viz
import numpy as np
import mne
import src.utilities.utilities_code as util
import os



class TestRSquaredBoxPlot:
    @pytest.fixture
    def set_up_data(self):
        modelType = ['Model_1', 'Model_1', 'Model_1','Model_2', 'Model_2', 'Model_2',
                     'Model_3', 'Model_3', 'Model_3']
        values = [1, 2, 3, 1, 2, 3, 1, 2, 3]
        data_frame = pd.DataFrame(data={'Model': modelType,'Value': values})
        yield data_frame


    @pytest.mark.mpl_image_compare
    def test_r_squared_boxplot(self,set_up_data, tmpdir):
        output_path = tmpdir.join('Output/')
        return viz.r_squared_boxplot(set_up_data, 'Model', 'Value', 'test', output_path)


class TestPlotRerp:

    @pytest.fixture
    def set_up_data(self):

        channel_names = ["A1","A2","A3","A4","A5","A6","A7","A8","A9","A10","A11","A12","A13","A14","A15","A16",
                    "A17","A18","A19","A20","A21","A22","A23","A24","A25","A26","A27","A28","A29","A30","A31","A32",
                    "B1","B2","B3","B4","B5","B6","B7","B8","B9","B10","B11","B12","B13","B14","B15","B16",
                    "B17","B18","B19","B20","B21","B22","B23","B24","B25","B26","B27","B28","B29","B30","B31","B32",
                    "C1","C2","C3","C4","C5","C6","C7","C8","C9","C10","C11","C12","C13","C14","C15","C16",
                    "C17","C18","C19","C20","C21","C22","C23","C24","C25","C26","C27","C28","C29","C30","C31","C32",
                    "D1","D2","D3","D4","D5","D6","D7","D8","D9","D10","D11","D12","D13","D14","D15","D16",
                    "D17","D18","D19","D20","D21","D22","D23","D24","D25","D26","D27","D28","D29","D30","D31","D32"]
        sampling_rate = 100
        channel_types = 'eeg'
        info = mne.create_info(channel_names ,sampling_rate, channel_types)
        np.random.seed(1)
        data = np.random.randn(10, 128, sampling_rate*2)

        data_evoked = data.mean(0)
        tmin = -0.1
        nave = data.shape[0]

        comment = "Smiley faces"

        # Create the Evoked object
        evoked_array = mne.EvokedArray(data_evoked, info, tmin,comment=comment, nave=nave)
        evoked_array.set_montage('biosemi128')
        Evoked_dictionary = {'Evoked_Array_1':evoked_array}
        yield Evoked_dictionary

    @pytest.mark.mpl_image_compare
    def test_plot_rERP(self,set_up_data, tmpdir):
        output_path = tmpdir.join('Output/')
        return viz.plot_rERP(set_up_data,output_path)

#TODO change to test data
class TestPlotRSquared:

    @pytest.fixture
    def set_up_data(self):
        frames = []
        for i in range(4):
            modelType = ['Model_1', 'Model_1', 'Model_1', 'Model_2', 'Model_2', 'Model_2',
                         'Model_2', 'Model_2', 'Model_2','Model_1', 'Model_1', 'Model_1',
                         'Model_2', 'Model_2', 'Model_2','Model_1', 'Model_1', 'Model_1']
            values = [1.0, 2.3, 3.5, 1.5, 2.3, 3.78, 1.7, 2.3, 4.7,1.0, 2.3, 3.5, 1.5, 2.3, 3.78, 1.7, 2.3, 4.7]
            data_frame = pd.DataFrame(data={'Model': modelType, 'Value': values})
            frames.append(data_frame)
        yield frames

    @pytest.mark.mpl_image_compare
    def test_plot_r_squared(self,set_up_data):
        frames = util.load_dataframes('Results/Reading_Data/Individual_subjects/R_Score_Eye_measure/', index_col=True)
        x = 'metric'
        y= 'Adjusted_R_Squared'
        sub_title = ['Measure1','Measure2','Measure3','Measure3']
        return viz.plot_R_multi_measure(frames,x,y,2,2,'Test Plot',sub_title)
















