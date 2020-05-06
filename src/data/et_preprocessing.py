import pandas as pd


class eye_tracking_preprocessing():

    #TODO add the fucntionality of selecting all the subjects in the data without specifying
    def _seperate_subject_data(self, data, subject_ids, id_column, folderpath=None):
        """ seperates data from subjects into individual dataframes returen in a list of datagrames

        Parameters
        ----------
        data : dataframe
            Dataframe containing the data from the subjects

        subject_ids: list
            List of subject ids that will be seperated into individual dataframs

        id_column: string
            The column name in the dataframe that contains the subject ids

        folderpath: string
            path to a folder that the individual subject dataframe will be saved to
        Returns
        -------
        individual_subject_frames : list
            list of dataframes corresponding to the ids in the subject_ids parameter

        """

        individual_subject_frames = []
        for i in subject_ids:
            subject_data = data[data[id_column] == i].copy()

            subject_data.drop(id_column, inplace=True, axis=1)

            subject_data = self.correct_null_data(subject_data)

            if (folderpath != None):
                subject_data.to_csv(folderpath + 'Subject_' + i + '.csv', index=False)

            individual_subject_frames.append(subject_data)

        return individual_subject_frames

    def aggregate_subject_data(self, data, subject_IDs, id_column, text_column):
        """ seperates data from subjects into individual dataframes returen in a list of datagrames

        Parameters
        ----------
        data : dataframe
            Dataframe containing the data from the subjects

        subject_ids: list
            List of subject ids that will be seperated into individual dataframs

        id_column: string
            The column name in the dataframe that contains the subject ids

        text_column: string
            The column name in the dataframe that contains the text data

        Returns
        -------
        average_subject_measures : Dataframe
            dataframe that has the average reading measure for each word for the subjects
            specified in the subject_ids parameter

        """

        # Seperate the data into the specified subjects
        subjects_reading_data = self._seperate_subject_data(data, subject_IDs, id_column)
        text = pd.DataFrame(subjects_reading_data[0][text_column])
        concat_df = []

        # drop the text column and make columns numeric
        for ind_subject_data in subjects_reading_data:
            ind_subject_data.drop([text_column], axis=1, inplace=True)
            ind_subject_data = ind_subject_data.apply(pd.to_numeric, errors='coerce')
            concat_df.append(ind_subject_data)

        summation_subject_data = sum(concat_df)
        average_subject_measures = summation_subject_data.div(len(subject_IDs))
        average_subject_measures = pd.concat([text, average_subject_measures], axis=1)

        return average_subject_measures


    def correct_null_data(self, data_frame):
        """ Replace data that is null and data that represents null

        Parameters
        ----------
        data_frame : dataframe
            Dataframe with that contains null data

        Returns
        -------
        null_corrected_dataframe : Dataframe
            dataframe with the null data imputed
        """

        data_frame.replace(".", 0, inplace=True)

        for column in data_frame.columns:
            if pd.api.types.is_string_dtype(data_frame[column]):
                data_frame[column] = data_frame[column].fillna('<unk>')
            else:
                data_frame[column] = data_frame[column].fillna(0)

        null_corrected_data_frame = data_frame.reset_index(drop=True)
        return null_corrected_data_frame


    def remove_outliers(self, data_frame, column_name):
        """ Remove data that is outside 2.5 sd of the mean of the column_name

        Parameters
        ----------
        data_frame : dataframe
            Dataframe with the reading data

        column_name: string
            The name of the column which will be used to calculate the
            outlier lower and upper threshold

        Returns
        -------
        data_frame : Dataframe
            dataframe whose rows fall below the lower and above the upper threshold removed

        """
        sd = data_frame.std()
        mean = data_frame.mean()
        lower = mean[column_name] - (sd[column_name] * 2.5)
        upper = mean[column_name] + (sd[column_name] * 2.5)
        remove_ET_outliers = (data_frame[column_name] >= lower) & (data_frame[column_name] <= upper)

        data_frame = data_frame[remove_ET_outliers]
        self.remaining_indexes = data_frame.index

        return data_frame

    def remove_lexical_processing(self, data_frame, column_name, threshold):

        """ Remove data that is outside 2.5 sd of the mean of the column_name

        Parameters
        ----------
        data_frame : dataframe
            Dataframe with the reading data

        column_name: string
            The name of the column which will be used to determine the lexical processing

        Returns
        -------
        data_frame : Dataframe
            dataframe whose rows fall below the lower the threshold are removed

        """

        condition = data_frame[column_name] > threshold
        data_frame = data_frame[condition]
        self.remaining_indexes = data_frame.index

        return data_frame

    def filtered_indicies(self, data):
        df = data.iloc[list(self.remaining_indexes)]
        return df


