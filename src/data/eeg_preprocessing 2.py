import mne
import numpy as np
import scipy.io as sio
from mne.preprocessing import ICA
from pyprep.noisy import Noisydata


class EEGPreprocessor:

    def __init__(self, sampling_rate, channel_labels, montage_name, filepath=None, raw=None):
        mne.set_log_level(verbose='ERROR')
        self.montage_name = montage_name
        self.montage = mne.channels.make_standard_montage(montage_name)
        self.channel_labels = channel_labels
        self.sampling_rate = sampling_rate
        self.raw, self.info = self.create_raw_object(self.sampling_rate, self.channel_labels, self.montage, raw_data=raw,filepath= filepath)

    def create_raw_object(self, sampling_rate, channel_labels, montage_name, raw_data=None, filepath=None):
        """ creates an mne raw object from the raw data of the eeg recordings

        Parameters
        ----------
        filepath : string
            string to the file that contains the raw data of the eeg recordings

        sampling_rate: integer
            integer to represent the frequency the eeg data was recorded at

        channel_labels: list
            list of strings corresponding to the channel names

        montage_name: string
            The configuration of the scalp used to record the eeg data

        Returns
        -------
        raw : mne.RawArray
            mne object that stores the values of the recording as well as the information
            about the eeg recording such as sampling rate, bad channels

        info: dictionary
            store information about the eeg recordings.

        Notes
        -----
        The raw object encapsulates the info object

        """
        if filepath != None:
            mat_contents = sio.loadmat(filepath)
            eegData = mat_contents['eegData']
            mastoids = mat_contents['mastoids']

            avg_mast = np.mean(mastoids, axis=1).reshape(len(mastoids), 1)
            eeg_data = np.subtract(eegData, avg_mast)
        else:
            eeg_data = raw_data


        eeg_data = eeg_data.T
        info = mne.create_info(ch_names=channel_labels, ch_types='eeg', sfreq=sampling_rate, verbose=None)
        raw = mne.io.RawArray(eeg_data, info, verbose='ERROR')
        raw.set_montage(montage_name)
        return raw, info

    def interpolate_bad_channels(self,raw=None, montage_name=None):
        """ detetcs the bad channels and interpolates them based on the good channels
        Parameters
        ----------
        raw : mne.rawArray
            raw object of the EEG recordings

        montage_name: string
            The configuration of the scalp used to record the eeg data

        Notes
        -----
        The classes interpolated_raw property is set with the data of the raw object
        that has been interpolated.
        """

        channel_detector = Noisydata(self.raw, self.montage_name, low_cut=1, high_cut=8)
        channel_detector.find_all_bads()

        # bad channels detected using five statistical figures
        bad_channels = list(set().union(
            *[channel_detector.bad_by_correlation, channel_detector.bad_by_deviation, channel_detector.bad_by_flat,
              channel_detector.bad_by_hf_noise, channel_detector.bad_by_nan]))

        filtered_data = channel_detector.raw_copy
        filtered_data.info['bads'].extend(bad_channels)
        eeg_interpolated = filtered_data.interpolate_bads(reset_bads=True)

        self.interpolated_raw = eeg_interpolated

    def perform_ICA(self, number_components,raw=None, plot=False):
        """ Performs independent component analysis to get rid of artefacts

        Parameters
        ----------
        raw : mne.rawArray
            raw object of the EEG recordings

        number of compentents: integer
            The number of components to return f in the ica anlaysis

        Notes
        -----
        Fifteen components is usually the recomended minimum of components analysis to perform
        """

        ica = ICA(n_components=number_components, random_state=97)

        if (self.interpolated_raw == None):
            ica.fit(self.raw)
            ica.apply(self.raw, exclude=[0])
        else:
            ica.fit(self.interpolated_raw)
            ica.apply(self.interpolated_raw, exclude=[0])

        if (plot == True):
            ica.plot_components()


    def read_events(self, filepath, event_type):
        """ Function to read the events from a matlab file and format for regression

        Parameters
        ----------
        filepath : string
            filepath to the file that contains the events of the eeg recordings

        event_type: string
            the event type to use in the anlysis

        Returns
        -----
        events: numpy array
            A 3 dimensional array that consists of event time, colun of zeros and event id
        """

        event_contents = sio.loadmat(filepath)
        onset_time = np.around(event_contents[event_type] * self.sampling_rate, decimals=0).astype(int)
        event_number = len(onset_time)
        zeros = np.zeros(event_number).reshape(event_number, 1)
        event_id = np.ones(event_number).reshape(event_number, 1)
        events = np.concatenate((onset_time, zeros, event_id), axis=1).astype(int)
        self.events = events
        return events


    def epoch_to_df(self, raw, events, event_id, tmin, tmax):
        """ Performs independent component analysis to get rid of artefacts

        Parameters
        ----------
        raw : mne.rawArray
            raw object of the EEG recordings

        events: numpy array
            The number of components to return f in the ica anlaysis

        event_id: dictionary
            dictionary of the event name and the number associated with it

        tmin: float
            The start time to consider from each evet time, can be a negative number

        tmax: float
            The end time to consider from each event time.

        Returns:
        -------
        epochs: dataframe
                The dataframe consist of all the epochs relating to the events in the recordings
                the index designates what epoch it is.

        """
        epochs = mne.Epochs(raw, events, event_id, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)
        index = ['epoch']
        df = epochs.to_data_frame(picks=None, index=index, scalings={'eeg': 1})
        self.epochs = df

    def preprocess_Data(self):
        """ A wrapper function to perfrom interpolation and ica analysis in one fuction call"""


        self.interpolate_bad_channels()
        self.perform_ICA(20)

