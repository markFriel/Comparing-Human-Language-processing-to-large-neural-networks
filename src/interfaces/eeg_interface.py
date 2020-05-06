from src.data.text_preprocessing import preprocessing_text
from src.features.language_modelling import languageModel
from src.data.sequence_preprocessing import sequence_processor
from src.data.eeg_preprocessing import EEGPreprocessor
from src.models.eeg_regression import TimeResolvedRegression
from sklearn.linear_model import  Ridge
import pandas as pd


def calculate_covariates(model_type, word_list, context=50):
    # Preprocess words for language modelling
    text_processor = preprocessing_text()
    clean_words = text_processor.clean_text(word_list)
    text = text_processor.word_list_to_string(clean_words)

    # Sequence for Language models
    sp = sequence_processor(model_type, text)
    sp.tokenizeWords()
    sp.text_to_sequences(context)

    # Language model computation of surprisal
    model = languageModel(model_type)
    model.word_probability(sp.indexed_sequences, sp.indexed_nextWord)
    model.clean_predicted_words(sp.indexed_tokens[0])

    if(model_type =='gpt2' or model_type=='gpt2xl'):
        model.collaspeSubWords(clean_words)

    model.computeSurprisal()

    covariate_name = 'Surprisal'
    covariates = pd.DataFrame(data={covariate_name: model.surprisal})

    return covariates


def preprocessed_eeg_raw(channel_name_path,eeg_path,montage, event_path, event_type):

    with open(channel_name_path) as f:
        channel_names = f.read().splitlines()
        del f
    eeg_object = EEGPreprocessor(128, channel_names, montage, filepath=eeg_path)
    eeg_object.preprocess_Data()
    raw = eeg_object.interpolated_raw

    events = eeg_object.read_events(event_path, event_type)

    return raw, events


def time_resolved_regression(raw,events,event_id,covariates):
    regression_data = TimeResolvedRegression(raw, events, event_id, covariates=covariates, tmin=-0.2, tmax=1)
    regression_data.raw = None
    rmodel = Ridge(alpha=1000, fit_intercept=False).fit(regression_data.design_matrix, regression_data.expected_data.T)
    coefficients = rmodel.coef_
    return coefficients, regression_data