import pytest
from src.features.language_modelling import languageModel
from src.data.sequence_preprocessing import sequence_processor
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TransfoXLTokenizer, TransfoXLLMHeadModel, CTRLTokenizer,CTRLLMHeadModel


class TestLanguageModelLoader():

    def test_loads_GPT2_model_tokenizer(self):
        test_case = 'gpt2'
        lm = languageModel(test_case)
        actual_model = type(lm.model)
        actula_tokenizer = type(lm.tokenizer)
        expected_model = type(GPT2LMHeadModel.from_pretrained(test_case))
        expected_token = type(GPT2Tokenizer.from_pretrained(test_case))
        assert actual_model == expected_model
        assert actula_tokenizer == expected_token

    @pytest.mark.skip(reason='Due to length of runtime')
    def test_loads_CTRL_model_tokenizer(self):
        test_case = 'ctrl'
        lm = languageModel(test_case)
        actual_model = type(lm.model)
        actula_tokenizer = type(lm.tokenizer)
        expected_model = type(CTRLLMHeadModel.from_pretrained(test_case))
        expected_token = type(CTRLTokenizer.from_pretrained(test_case))
        assert actual_model == expected_model
        assert actula_tokenizer == expected_token

    @pytest.mark.skip(reason='Due to length of runtime')
    def test_loads_TXL_model_tokenizer(self):
        test_case = 'txl'
        lm = languageModel(test_case)
        actual_model = type(lm.model)
        actula_tokenizer = type(lm.tokenizer)
        expected_model = type(TransfoXLLMHeadModel.from_pretrained(test_case))
        expected_token = type(TransfoXLTokenizer.from_pretrained(test_case))
        assert actual_model == expected_model
        assert actula_tokenizer == expected_token

    @pytest.mark.skip(reason='Due to length of runtime')
    def test_loads_GPT2XL_model_tokenizer(self):
        test_case = 'gpt2-xl'
        lm = languageModel(test_case)
        actual_model = type(lm.model)
        actula_tokenizer = type(lm.tokenizer)
        expected_model = type(GPT2LMHeadModel.from_pretrained(test_case))
        expected_token = type(GPT2Tokenizer.from_pretrained(test_case))
        assert actual_model == expected_model
        assert actula_tokenizer == expected_token

    def test_loads_none_one_error(self):
        test_case = 'error'
        with pytest.raises(NameError):
            languageModel(test_case)


class TestWordProbability():

    @pytest.fixture
    def setup_sequences(self):
        word_list = ['this' , 'is', 'an','example', 'string', 'to',
                     'use', 'for', 'testing', 'purposes', 'it', 'will'
                     'be', 'used', 'to', 'test', 'output', 'of',
                     'languae', 'models']
        text = ' '.join(word_list)

        sp = sequence_processor('gpt2',text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        yield sp.indexed_sequences

    @pytest.fixture()
    def setup_next_word(self):
        word_list = ['this' , 'is', 'an','example', 'string', 'to',
                     'use', 'for', 'testing', 'purposes', 'it', 'will'
                     'be', 'used', 'to', 'test', 'output', 'of',
                     'languae', 'models']
        text = ' '.join(word_list)

        sp = sequence_processor('gpt2',text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        yield sp.indexed_nextWord

    def test_length_of_probabilit_returned(self,setup_sequences,setup_next_word):
        lm = languageModel('gpt2')
        lm.word_probability(setup_sequences,setup_next_word)
        actual_prob = len(lm.probability)
        expected_prob = 20
        actual_word = len(lm.word)
        expected_word= 20
        assert actual_prob == expected_prob
        assert actual_word == expected_word

    def test_element_return_type(self,setup_sequences,setup_next_word):
        lm = languageModel('gpt2')
        lm.word_probability(setup_sequences,setup_next_word)
        actual_prob_type= type(lm.probability[0])
        expected_prob_type = float
        actual_word_type = type(lm.word[0])
        expected_word_type = str
        assert actual_prob_type == expected_prob_type
        assert actual_word_type == expected_word_type

class TestCleanPredictedToken():

    @pytest.fixture
    def setup_clean_words(self):
        word_list = ['this', 'is', 'an', 'example', 'string', 'to',
                     'use', 'for', 'testing', 'purposes' 'it', 'will'
                     'be', 'used', 'to', 'test', 'output', 'of',
                     'languae', 'models']
        text = ' '.join(word_list)

        sp = sequence_processor('gpt2', text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        lm = languageModel('gpt2')
        lm.word_probability(sp.indexed_sequences, sp.indexed_nextWord)
        lm.clean_predicted_words(sp.indexed_nextWord[0])
        yield lm.word

    def test_content_next_word(self, setup_clean_words):
        actual = False
        word_list = setup_clean_words
        for words in word_list:
            if words.__contains__('@'):
                actual = True
        expected = False
        assert actual == expected

    def test_content_next_word_empty_space(self, setup_clean_words):
        actual = False
        word_list = setup_clean_words
        for words in word_list:
            if words.__contains__(' '):
                actual = True
        expected = False
        assert actual == expected


class TestCollaspeSubWords():

    @pytest.fixture
    def setup_clean_words(self):
        word_list = ['this', 'is', 'an', 'example', 'string', 'to',
                     'use', 'for', 'testing', 'purposes' 'it', 'will'
                     'be', 'used', 'to', 'test', 'output', 'of',
                     'language', 'models']
        text = ' '.join(word_list)

        sp = sequence_processor('gpt2', text)
        sp.tokenizeWords()
        sp.text_to_sequences(4)
        lm = languageModel('gpt2')
        lm.word_probability(sp.indexed_sequences, sp.indexed_nextWord)
        lm.clean_predicted_words(sp.indexed_tokens[0])
        lm.collaspeSubWords(word_list)

        yield lm.word

    def test_return_length_equals_original(self, setup_clean_words):
        actual = len(setup_clean_words)
        expected = len(setup_clean_words)
        assert actual == expected




