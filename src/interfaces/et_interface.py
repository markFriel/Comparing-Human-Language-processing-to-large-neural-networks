from src.data.text_preprocessing import preprocessing_text
from src.features.language_modelling import languageModel
from src.data.sequence_preprocessing import sequence_processor
from src.data.et_preprocessing import eye_tracking_preprocessing

def preprocess_reading_data(reading_data, text_column, metric_column, threshold):
    """ Process the eye-tracking measures and return the words corresponding to them

    Parameters
    ----------
    reading_data : Dataframe
        dataframe containg reading measures of the words

    text_column: str
        name of the column containg the words

    metric_column: str
        name of the colum to base the outlier removal on

    threashold: int
        number to remove those rows who fall below this threshold

    Returns
    -------
    eye_track : dataframe
        a data frame containg the eye-mesure values

    word_list: list
        list of words
    """

    data_processor = eye_tracking_preprocessing()
    sub_reading_data = data_processor.remove_outliers(reading_data, metric_column)
    sub_reading_data = data_processor.remove_lexical_processing(sub_reading_data, metric_column, threshold)
    word_list = list(sub_reading_data[text_column])
    eye_track = sub_reading_data.iloc[:, 1:]

    #eye_track.columns = ['First Fixation Duration', 'Gaze Duration', 'Total Reading Time', 'Word Skip']

    return eye_track, word_list


def calculate_covariates(model_type, word_list, context=50):
    """ Removes non-ascii characters from words in word list

        Parameters
        ----------
        model_type : str
            language model code to select the model to calculate surprisal

        word_list: list
            list of words for surprisal to be calculated for

        context: int
            the amount of words to be consided as context for caculating surprisal

        Returns
        -------
        model_surprisal : list
            surprisal values for each word

        word_frequence: list
            list of the number of occurence for each word

        word_length: list
            list of word lengths
        """

    # Preprocess words for language modelling

    text_processor = preprocessing_text()
    clean_words = text_processor.clean_text(word_list)
    text = text_processor.word_list_to_string(clean_words)


    # Sequence for Language models
    sp = sequence_processor(model_type, text)
    sp.tokenizeWords()
    sp.text_to_sequences(context)
    sp.word_frequency_length()
    word_length = sp.word_lengths
    word_frequency = sp.word_frequencies

    # Language model commputation of surprisal
    model = languageModel(model_type)
    model.word_probability(sp.indexed_sequences, sp.indexed_nextWord)
    model.clean_predicted_words(sp.indexed_tokens[0])

    if (model_type == 'gpt2' or model_type == 'gpt2xl'):
        model.collaspeSubWords(clean_words)

    model.computeSurprisal()

    return model.surprisal, word_length, word_frequency


