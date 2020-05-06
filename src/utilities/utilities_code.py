import pandas as pd
import os
from src.data.eeg_preprocessing import EEGPreprocessor
from mne import concatenate_raws
import os

def read_flaten_word_list(filepath):
    text_data = pd.read_csv(filepath, header=None)
    lists_word_list = text_data.values.tolist()
    word_list = [item for sublist in lists_word_list for item in sublist]
    return word_list


def read_directory(file_path):
    files = os.listdir(file_path)
    if files.__contains__('.DS_Store'):
        files.remove('.DS_Store')
    if files.__contains__('.ipynb_checkpoints'):
        files.remove('.ipynb_checkpoints')
    return files


def load_dataframes(foldepath, index_col=True):
    files = read_directory(foldepath)
    frames =[]
    for file in files:

        filePath = foldepath + file

        if(index_col==True):
            df = pd.read_csv(filePath, index_col=0)
        else:
            df = pd.read_csv(filePath)
        frames.append(df)

    return frames




def concatenate_covariates(folderpath, model_type, pop_file=None):
    files = sorted(read_directory(folderpath))

    if pop_file is not None:
        files.remove(pop_file)
    concat_covs=[]
    for i in files:
        df =pd.DataFrame(pd.read_csv(folderpath + i)[model_type])
        concat_covs.append(df)
    c_covs  = pd.concat(concat_covs, axis=0, ignore_index=True)
    return c_covs


def concatenate_runs(eeg_path, event_path, channel_names, channel_montage,event_type,last_time=0):
    concat_events = []
    concat_raws = []

    runs = sorted(read_directory(eeg_path))
    events = sorted(read_directory(event_path))


    for i , file in enumerate(runs):
        eeg_object = EEGPreprocessor(128, channel_names, channel_montage, filepath=eeg_path + runs[i])
        eeg_object.preprocess_Data()
        event = pd.DataFrame(eeg_object.read_events(event_path+ events[i], event_type))

        event.iloc[:][0] = event.iloc[:][0] + last_time
        concat_events.append(event)

        concat_raws.append(eeg_object.interpolated_raw)
        last_time += eeg_object.interpolated_raw._data.shape[1]


    c_events = pd.concat(concat_events, axis=0, ignore_index=True)
    c_events = c_events.values
    eeg_data = concatenate_raws(concat_raws)

    return eeg_data, c_events

def concatenate_subjects(eeg_path, event_path, channel_names, channel_montage,event_type):
    sub_raws =[]
    sub_events =[]

    subject = sorted(read_directory(eeg_path))

    last_time =0
    for i in subject:
        data_path = eeg_path + i +'/'
        raw, events  = concatenate_runs(data_path, event_path, channel_names, channel_montage,event_type,last_time)
        last_time += raw._data.shape[1]
        events = pd.DataFrame(events)
        sub_raws.append(raw)
        sub_events.append(events)
    c_events = pd.concat(sub_events, axis=0, ignore_index=True)
    c_events = c_events.values
    c_raws = concatenate_raws(sub_raws)
    return c_raws, c_events
