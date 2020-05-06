import pytest
import numpy as np
import scipy.io as sio
import scipy.io.matlab.mio5
from src.data.eeg_preprocessing import EEGPreprocessor
import mne
import pandas as pd
import os

class TestCreateRawObject():

    @pytest.fixture
    def create_numpy_raw_data(self):
        raw_data_array = np.random.randint(1,10,(1280,128))
        yield raw_data_array

    @pytest.fixture
    def create_channel_names(self, tmpdir):
        channel_name_filepath = tmpdir.join('channel_names.txt')
        with open(channel_name_filepath, "w") as f:
            f.write("A1\n""A2\n""A3\n""A4\n""A5\n""A6\n""A7\n""A8\n""A9\n""A10\n""A11\n""A12\n""A13\n""A14\n""A15\n""A16\n"
                    "A17\n""A18\n""A19\n""A20\n""A21\n""A22\n""A23\n""A24\n""A25\n""A26\n""A27\n""A28\n""A29\n""A30\n""A31\n""A32\n"
                    "B1\n""B2\n""B3\n""B4\n""B5\n""B6\n""B7\n""B8\n""B9\n""B10\n""B11\n""B12\n""B13\n""B14\n""B15\n""B16\n"
                    "B17\n""B18\n""B19\n""B20\n""B21\n""B22\n""B23\n""B24\n""B25\n""B26\n""B27\n""B28\n""B29\n""B30\n""B31\n""B32\n"
                    "C1\n""C2\n""C3\n""C4\n""C5\n""C6\n""C7\n""C8\n""C9\n""C10\n""C11\n""C12\n""C13\n""C14\n""C15\n""C16\n"
                    "C17\n""C18\n""C19\n""C20\n""C21\n""C22\n""C23\n""C24\n""C25\n""C26\n""C27\n""C28\n""C29\n""C30\n""C31\n""C32\n"
                    "D1\n""D2\n""D3\n""D4\n""D5\n""D6\n""D7\n""D8\n""D9\n""D10\n""D11\n""D12\n""D13\n""D14\n""D15\n""D16\n"
                    "D17\n""D18\n""D19\n""D20\n""D21\n""D22\n""D23\n""D24\n""D25\n""D26\n""D27\n""D28\n""D29\n""D30\n""D31\n""D32\n"
                    )

        yield channel_name_filepath

    def test_returns_correct_shape(self,create_numpy_raw_data,create_channel_names):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f

        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        raw = processor.raw
        expected = (128,1280)
        actual =  raw._data.shape
        assert actual == expected


    def test_returns_an_object_of_raw_array(self,create_numpy_raw_data,create_channel_names):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f

        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        expected_raw = mne.io.array.array.RawArray
        actual_raw =  type(processor.raw)
        expected_info = mne.io.meas_info.Info
        actual_info = type(processor.info)
        assert actual_raw == expected_raw
        assert actual_info == expected_info

    def test_montage_is_correct(self,create_numpy_raw_data,create_channel_names):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f

        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        actual = type(processor.montage)
        expected = mne.channels.montage.DigMontage
        assert actual == expected

class TestInterpolateBadChannels():

    @pytest.fixture
    def create_numpy_raw_data(self):
        raw_data_array = np.random.randint(1, 2, (1280, 128))
        yield raw_data_array

    @pytest.fixture
    def create_channel_names(self, tmpdir):
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

        yield channel_name_filepath

    @pytest.mark.xfail
    def test_return_of_interpolate_bad_channels(self,create_channel_names, create_numpy_raw_data):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f
        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        processor.interpolate_bad_channels()
        actual =type(processor.interpolated_raw)
        expected = mne.io.array.array.RawArray
        assert actual == expected


class TestDfToEpochs():

    @pytest.fixture
    def create_numpy_raw_data(self):
        raw_data_array = np.random.randint(1, 2, (1280, 128))
        yield raw_data_array

    @pytest.fixture
    def create_channel_names(self, tmpdir):
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

        yield channel_name_filepath

    @pytest.fixture
    def create_events(self):
        event_times = sorted(np.random.choice(range(1280), 50, replace=False))
        event_column = np.zeros(50).astype(int)
        event_id = np.ones(50).astype(int)
        events = pd.DataFrame(data={'Times': event_times, 'Blank':event_column, 'Event_type':event_id})
        yield events


    def test_return_events(self,create_numpy_raw_data,create_channel_names, create_events):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f
        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        event_id ={'Event_id':1}
        processor.epoch_to_df(processor.raw,create_events,event_id,-0.1,.5)
        epochs = processor.epochs
        actual = type(epochs)
        expected = pd.DataFrame
        assert actual == expected

class TestReadEvents():

    @pytest.fixture
    def create_numpy_raw_data(self):
        raw_data_array = np.random.randint(1, 2, (1280, 128))
        yield raw_data_array

    @pytest.fixture
    def create_channel_names(self, tmpdir):
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

        yield channel_name_filepath

    @pytest.fixture
    def create_events(self):
        event_file_path = 'Events'
        event_times = sorted(np.random.choice(range(1280), 50, replace=False))
        event_column = np.zeros(50).astype(int)
        event_id = np.ones(50).astype(int)
        events ={'Times': event_times, 'Blank': event_column, 'Event_type': event_id}
        sio.savemat(event_file_path, events, appendmat=True)
        yield event_file_path



    def test_read_events_return(self,create_numpy_raw_data,create_channel_names,create_events):
        with open(create_channel_names) as f:
            channel_names = f.read().splitlines()
            del f
        processor = EEGPreprocessor(128, channel_names, 'biosemi128', raw=create_numpy_raw_data)
        print(create_events)
        event_id ={'Event_id':1}
        processor.read_events(create_events,'Event_type')
        actual = type(processor.events)
        expected = np.ndarray
        assert actual == expected
        os.remove(create_events)









