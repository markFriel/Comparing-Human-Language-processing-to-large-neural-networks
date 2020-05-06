import pytest
import src.configurations.EEG_Run_configuration as eec
import mne
import numpy as np
from scipy.io import savemat
from random_words import RandomWords
import os
from os import path
import pandas as pd

class TestSingleRun:

    @pytest.fixture
    def set_up_data(self, tmpdir):
        data = np.random.randint(1, 1000, (1000, 128))
        mastoids = np.random.randint(1, 1000, (1000, 2))
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

        dictionary_variables = {'eegData': data, 'mastoids': mastoids}
        savemat(filepath, dictionary_variables)

        event_filepath = 'Output/event_file.mat'
        events = np.array([[0], [100], [200], [300], [400], [500], [600], [700], [800]])
        event_dictionary = {'onset_time': events}
        savemat(event_filepath, event_dictionary)

        text_name_filepath = tmpdir.join('text_names.txt')
        with open(text_name_filepath, "w") as f:
            f.write("This\n""is\n""a\n""test\n""sentence\n""for\n""testing\n""the\n""integration\n")

        yield filepath,text_name_filepath, channel_name_filepath, event_filepath
        os.remove(filepath)
        os.remove(event_filepath)

    def test_file_presence(self, set_up_data):
        filepath, text_name_filepath, channel_name_filepath, event_filepath = set_up_data
        eec.single_run('gpt2', 50, text_name_filepath, channel_name_filepath, filepath,
                       event_filepath,'onset_time','biosemi128', 'Output/')

        plot_path = 'Output/' + 'onset_time.png'
        actual_onset = path.exists(plot_path)
        assert actual_onset == True
        os.remove(plot_path)

        plot_path = 'Output/' + 'Surprisal.png'
        actual_surprisal = path.exists(plot_path)
        assert actual_surprisal == True
        os.remove(plot_path)

