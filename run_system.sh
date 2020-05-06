#!/bin/bash

source venv_/bin/activate
python main.py 'EEG' 'eeg' 'sr' 'gpt2' 10 'EEG_Data/Word_Lists/Run_1.csv' 'EEG_Data/Channel_labels.txt' 'EEG_Data/EEG_recordings/Subject1/Subject1_Run1.mat' 'EEG_Data/Events/Run1.mat' 'onset_time' 'biosemi128' 'Output/'


#'Eye_track' 'eye_track' 'agg' 5 'gpt2' 'ET_Data/reading_data.csv' 'Output/'
