import argparse
import sys
import pandas as pd

import src.configurations.EEG_Run_configuration as erc
import src.configurations.ET_Run_configuration as etrc


if __name__ == '__main__':

    # Parent parser
    parent_parser = argparse.ArgumentParser()
    parent_parser.add_argument('analysis_type', help='EEG or Eye-track')
    subparsers = parent_parser.add_subparsers(help='sub-command help')

    # EEG Parser
    eeg_parser = subparsers.add_parser('eeg', help='Analysis for EEG data')
    eeg_parser.add_argument("Configuration_type", help="The run configuration used for analysis: sr | ss | cs")
    eeg_parser.add_argument("Model", help="The language model code to compute surprisal values")
    eeg_parser.add_argument("Context_length", type=int, help="The number of preceding words to consider when computing surprisal")
    eeg_parser.add_argument("Text_path", help="File path to the text data")
    eeg_parser.add_argument("Channel_path", help="Filepeath to channel name file")
    eeg_parser.add_argument('EEG_path', help="File path to the eeg data")
    eeg_parser.add_argument('Event_path', help="File path to the event data")
    eeg_parser.add_argument('Event_type', help="Event type")
    eeg_parser.add_argument('Montage', help="The montage type of the eeg scalp recordings")
    eeg_parser.add_argument('Output_file', help="Path to the folder for the analysis output")

    # ET Parser
    et_parser = subparsers.add_parser('eye_track', help='Analysis for Eye-tracking data')
    et_parser.add_argument("Configuration_type", help="The run configuration used for analysis: agg | indv")
    et_parser.add_argument("Context_length", type=int, help="The number of preceding words to consider when computing surprisal")
    et_parser.add_argument("Model", help="The model to run for the anlaysis")
    et_parser.add_argument("Reading_filepath", help="The path to the reading data")
    et_parser.add_argument("Output_folder", help="Path to the folder for the analysis output")
    args = parent_parser.parse_args()

    # ['eye_track', 'eye_track' ,'agg', 'gpt2', 'ET_Data/reading_data.csv', 'Output/']
    analysis_type = args.analysis_type

    if(analysis_type == 'EEG' ):

        # EEG Arguments
        run_type = args.Configuration_type
        model_type = args.Model
        context = args.Context_length
        text_filepath = args.Text_path
        channel_name_path = args.Channel_path
        eeg_path = args.EEG_path
        event_path = args.Event_path
        event_type = args.Event_type
        montage = args.Montage
        output_file = args.Output_file

        if (run_type == 'sr'):
            erc.single_run(model_type, context, text_filepath, channel_name_path, eeg_path,
                           event_path, event_type, montage, output_file)

        elif (run_type == 'ss'):
            erc.single_subject(model_type, context, text_filepath, channel_name_path, eeg_path,
                               event_path, event_type, montage, output_file)

        elif (run_type == 'cs'):
            erc.cross_subject(model_type, context, text_filepath, channel_name_path, eeg_path,
                              event_path, event_type, montage, output_file)

        else:
            print('Run configuration is not valid')
            sys.exit()

    elif(analysis_type == 'Eye_track'):

        run_type = args.Configuration_type
        context = args.Context_length
        model_type = args.Model
        reading_filepath = args.Reading_filepath
        output_file = args.Output_folder

        readingData = pd.read_csv(reading_filepath)
        sub_readingData = readingData[
            ['PP_NR', 'WORD', 'WORD_GAZE_DURATION', 'WORD_FIRST_FIXATION_DURATION', 'WORD_TOTAL_READING_TIME',
             'WORD_SKIP']]
        subject_IDs = ['pp21', 'pp22', 'pp23', 'pp26', 'pp29', 'pp30', 'pp31', 'pp32', 'pp33', 'pp34', 'pp35']
        subject_column = 'PP_NR'
        text_column = 'WORD'
        metric_column = 'WORD_FIRST_FIXATION_DURATION'

        if (run_type == 'agg'):
            etrc.aggregated_data_analysis(model_type,context, sub_readingData, subject_IDs, subject_column,
                                          text_column, metric_column, 100, output_file)

        elif (run_type == 'indv'):
            etrc.individual_data_analysis(model_type,context, sub_readingData, subject_IDs, subject_column,
                                          text_column, metric_column, 100, output_file)

        else:
            print('Run configuration is not valid')
            sys.exit()

