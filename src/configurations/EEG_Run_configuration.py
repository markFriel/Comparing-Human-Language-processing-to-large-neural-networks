import src.visualisation.visualisation as ev
import src.models.eeg_regression as trr
import src.interfaces.eeg_interface as eec
import src.utilities.utilities_code as util
import numpy as np
import pandas as pd
import os

#TODO Refact so that each function hierarchy uses the method below it

def single_run(model_type, context, text_filepath,channel_name_path,eeg_path, event_path, event_type, montage, output_file):
    """ This is the configuration function for anlysing EEG data for a single recording
        Parameters
        ----------
        model_type : string
            A string to specify which language model to compute the word surprisal

        context: int
            the number of word to consider in the calculation of the probability of a next word

        text_filepath: basestring
            filepath to the file that contains the words whose surprisal values will be calculated

        channel_name_path: basestring
            filepath to file that contains the names of the channels in the head scalp montage

        eeg_path: basestring
            filepath to the file that contains the raw EEG data

        event_path: basestring
            filepath to the file that contains the event type and time

        event_type: basestring
            denotes what event type tp use in the analysis

        montage: basestring
            the head scalp layout used to record the eeg data

        output_file: basestring
            filepath to the output folder that the output of the anlaysis will be saved to



        Notes
        -----
        This function creates two plots of the coefficients for word-onset/offset and surprisal
        """
    event_id = {event_type: 1}

    # calculate word Surprisal
    word_list = util.read_flaten_word_list(text_filepath)
    covariates = eec.calculate_covariates(model_type, word_list, context)
    # preprocess of eeg data
    raw, events = eec.preprocessed_eeg_raw(channel_name_path,eeg_path,montage,event_path,event_type)

    # Time Resolved Regression
    coefficients, regression_data = eec.time_resolved_regression(raw,events,event_id, covariates)
    Evoked = trr._make_evokeds(coefficients, regression_data.conds, regression_data.cond_length, regression_data.tmin_s,regression_data.tmax_s, regression_data.info)
    _ = ev.plot_rERP(Evoked, output_file)


def single_subject(model_type, context, text_filepath, channel_name_path, eeg_path, event_path, event_type, montage, output_file):
    """ This is the configuration function for anlysing EEG data for all the runs of a single subject
        Parameters
        ----------
        model_type : string
            A string to specify which language model to compute the word surprisal

        context: int
            the number of word to consider in the calculation of the probability of a next word

        text_filepath: basestring
            filepath to the file that contains the words whose surprisal values will be calculated

        channel_name_path: basestring
            filepath to file that contains the names of the channels in the head scalp montage

        eeg_path: basestring
            filepath to the file that contains the raw EEG data

        event_path: basestring
            filepath to the file that contains the event type and time

        event_type: basestring
            denotes what event type tp use in the analysis

        montage: basestring
            the head scalp layout used to record the eeg data

        output_file: basestring
            filepath to the output folder that the output of the anlaysis will be saved to

        Notes
        -----
        This function creates two plots of the coefficients for word-onset/offset and surprisal
        """
    event_id = {event_type: 1}
    coefficients= np.zeros((128,310))

    subject_Runs = sorted(util.read_directory(eeg_path))
    covariates_folder = sorted(util.read_directory(text_filepath))
    events_folder = sorted(util.read_directory(event_path))


    for i, run in enumerate(subject_Runs):
        # calculate word Surprisal
        cov_file = text_filepath + covariates_folder[i]
        word_list = util.read_flaten_word_list(cov_file)
        covariates = eec.calculate_covariates(model_type, word_list, context)

        #Preprocess the EEG data
        run_path = eeg_path + run
        event_file = event_path + events_folder[i]
        raw, events = eec.preprocessed_eeg_raw(channel_name_path, run_path, montage, event_file, event_type)

        # Create temporal design matrix and perform time resolved regression
        run_coefficients, regression_data = eec.time_resolved_regression(raw,events,event_id, covariates)
        coefficients += run_coefficients


    Evoked = trr._make_evokeds(coefficients, regression_data.conds, regression_data.cond_length, regression_data.tmin_s,regression_data.tmax_s, regression_data.info)
    _ = ev.plot_rERP(Evoked, output_file)


def cross_subject(model_type, context, text_filepath,channel_name_path,eeg_path, event_path, event_type, montage, output_file):
    """ This is the configuration function for anlysing EEG data for all the runs in the study
        Parameters
        ----------
        model_type : string
            A string to specify which language model to compute the word surprisal

        context: int
            the number of word to consider in the calculation of the probability of a next word

        text_filepath: basestring
            filepath to the file that contains the words whose surprisal values will be calculated

        channel_name_path: basestring
            filepath to file that contains the names of the channels in the head scalp montage

        eeg_path: basestring
            filepath to the file that contains the raw EEG data

        event_path: basestring
            filepath to the file that contains the event type and time

        event_type: basestring
            denotes what event type tp use in the analysis

        montage: basestring
            the head scalp layout used to record the eeg data

        output_file: basestring
            filepath to the output folder that the output of the anlaysis will be saved to



        Notes
        -----
        This function creates two plots of the coefficients for word-onset/offset and surprisal
        """

    event_id = {event_type: 1}
    coefficients= np.zeros((128,310))

    All_Test_Subjects = util.read_directory(eeg_path)


    for subject in All_Test_Subjects:
        Subject_Path = eeg_path + subject + '/'

        subject_Runs = sorted(util.read_directory(Subject_Path))
        events_folder = sorted(util.read_directory(event_path))
        covariates_folder = sorted(util.read_directory(text_filepath))


        for i, run in enumerate(subject_Runs):
            cov_file = text_filepath + covariates_folder[i]
            covariates = eec.calculate_covariates(model_type, cov_file, context)

            run_path = Subject_Path + run
            event_file = event_path + events_folder[i]
            raw, events = eec.preprocessed_eeg_raw(channel_name_path, run_path, montage, event_file, event_type)

            run_coefficients, regression_data = eec.time_resolved_regression(raw, events, event_id, covariates)
            coefficients += run_coefficients


    Evoked = trr._make_evokeds(coefficients, regression_data.conds, regression_data.cond_length,regression_data.tmin_s,regression_data.tmax_s, regression_data.info)
    _ = ev.plot_rERP(Evoked, output_file)