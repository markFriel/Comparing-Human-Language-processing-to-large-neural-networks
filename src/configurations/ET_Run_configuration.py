import src.models.et_regression as etr
from src.data.et_preprocessing import eye_tracking_preprocessing
import src.interfaces.et_interface as eti
import pandas as pd
import src.visualisation.visualisation as etv


def aggregated_data_analysis(model_type,context, reading_data, subject_IDs, subject_column,
                             text_column,  metric_column, threshold, output_file):

    """ The configuration function for the eye-tracking data when eye inspections are averaged
            Parameters
            ----------
            model_type : string
            A string to specify which language model to compute the word surprisal

            context: int
            the number of word to consider in the calculation of the probability of a next word

            reading_data: basestring
            filepath to the file that contains the eye-inspections and words in the text corpus

            subject_IDs: list
                the list of subject IDs whose data will be included in the analysis

            subject_column: basestring
                string that denotes the name of the column that denotes the subjects IDs

            text_column: basestring
                string that denotes the column that conatisn the words in the analysis

            metric_column: basestring
                string that denotes the name of the column that the rows will be filter upon

            threshold: integer
                this will set the value at which eye-inspections has to be over to remain in the anlysis

            output_file: basestring
                filepath to the folder that will contain the output table

            Notes
            -----
            This function outputs a table containing the adjusted R squared score for the base model
            and the model that includes surprisal, anf the difference between the two models
            """

    # Preprocess the eye-tracking data
    data_processor = eye_tracking_preprocessing()
    sub_reading_data = data_processor.aggregate_subject_data(reading_data, subject_IDs, subject_column, text_column)


    eye_tracking_measures, word_list = eti.preprocess_reading_data(sub_reading_data, text_column, metric_column,threshold)

    # Preprocess words for language modelling
    Surprisal, word_length, word_frequency = eti.calculate_covariates(model_type, word_list[:100], context)

    base_model = pd.DataFrame(data={'Length': word_length, 'Frequency': word_frequency})
    Surprisal_model = pd.DataFrame(data={'Surprisal': Surprisal, 'Length': word_length, 'Frequency': word_frequency})


    base_r_score = etr.K_fold_validation(5, base_model, eye_tracking_measures.iloc[:100,:])
    Surprisal_r_score = etr.K_fold_validation(5, Surprisal_model, eye_tracking_measures.iloc[:100,:])


    difference = [surprisal - base for surprisal, base in zip(Surprisal_r_score, base_r_score)]



    model_r_scores = pd.DataFrame(
        data={'Base Model': base_r_score, (model_type + ' Model'): Surprisal_r_score, 'Difference': difference}).T
    model_r_scores.columns = eye_tracking_measures.columns
    model_r_scores.to_csv(output_file + model_type + '.csv', index=True)



def individual_data_analysis(model_type, context, sub_reading_data, subject_IDs,subject_column,
                             text_column, metric_column, threshold, output_file):
    """ The configuration function for the eye-tracking data against the individual subjects inspections
        Parameters
        ----------
        model_type : string
        A string to specify which language model to compute the word surprisal

        context: int
        the number of word to consider in the calculation of the probability of a next word

        reading_data: basestring
        filepath to the file that contains the eye-inspections and words in the text corpus

        subject_IDs: list
            the list of subject IDs whose data will be included in the analysis

        subject_column: basestring
            string that denotes the name of the column that denotes the subjects IDs

        text_column: basestring
            string that denotes the column that conatisn the words in the analysis

        metric_column: basestring
            string that denotes the name of the column that the rows will be filter upon

        threshold: integer
            this will set the value at which eye-inspections has to be over to remain in the anlysis

        output_file: basestring
            filepath to the folder that will contain the output table

        Notes
        -----
        This function outputs a boxplot of the adjusted R squared score for the base model
        and a boxplot for model that includes surprisal.
        """
    base_df = []
    surprisal_df = []

    dp = eye_tracking_preprocessing()
    subject_reading_measure = dp._seperate_subject_data(sub_reading_data, subject_IDs, subject_column)

    for d_frame in subject_reading_measure:
        frame = d_frame.apply(pd.to_numeric, errors='ignore')

        eye_tracking_measures, word_list = eti.preprocess_reading_data(frame, text_column,metric_column,threshold)

        Surprisal, word_length, word_frequency = eti.calculate_covariates(model_type, word_list,context)

        base_model = pd.DataFrame(data={'Length': word_length, 'Frequency': word_frequency})
        Surprisal_model = pd.DataFrame(
            data={'Surprisal': Surprisal, 'Length': word_length, 'Frequency': word_frequency})

        base_r_score = etr.K_fold_validation(5, base_model, eye_tracking_measures)
        Surprisal_r_score = etr.K_fold_validation(5, Surprisal_model, eye_tracking_measures)

        base_df.append(base_r_score)
        surprisal_df.append(Surprisal_r_score)

    base_model_df = pd.DataFrame.from_records(base_df, columns=eye_tracking_measures.columns)
    surprisal_mode_df = pd.DataFrame.from_records(surprisal_df, columns=eye_tracking_measures.columns)

    for column in base_model_df.columns:
        result_df = pd.DataFrame(data={'Base': base_model_df[column], 'Surprisal': surprisal_mode_df[column]})
        df = etr.flatten_df(result_df)
        _ = etv.r_squared_boxplot(df, 'Metric', 'R_squared', column, output_file)