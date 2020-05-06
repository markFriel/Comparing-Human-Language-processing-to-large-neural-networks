import pytest
from src.data.text_preprocessing import preprocessing_text


class TestRemovePunctuation():

    def test_no_puntuation_remains(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
        actual = text_preprocessor.remove_puntuation(word_list)
        expected = ['  In', 'the ', 'beginning',' I', 'didnt', 'like', ' vegetables', 'Will ', 'I', ' ever', 'like ', 'them', '']
        assert actual == expected

    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.remove_puntuation(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list =[['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.remove_puntuation(word_list)

    def test_return_instance_of_list(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
        actual = text_preprocessor.remove_puntuation(word_list)
        assert isinstance(actual,list)


class TestWordToLower():

    def test_no_upper_case_letter_remain(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
        actual = text_preprocessor.words_to_lower(word_list)
        expected = ['  in', 'the ', 'beginning,', ' i', 'didn\'t', 'like', ' vegetables!', 'will ', 'i', ' ever','like ', 'them?', '!']
        assert actual == expected


    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.words_to_lower(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list = [['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.words_to_lower(word_list)

    def test_return_instance_of_list(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
        actual = text_preprocessor.words_to_lower(word_list)
        assert isinstance(actual,list)

class TestRemoveEmptySpace():

    def test_no_empty_strings(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', ' ', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ','like ', 'them?', '!']
        actual = text_preprocessor.remove_empty_space(word_list)
        expected = ['In', 'the','<unk>',  'I', 'didn\'t', 'like', 'vegetables!', 'Will', 'I','<unk>', 'like', 'them?', '!']
        assert actual == expected

    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.remove_empty_space(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list = [['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.remove_empty_space(word_list)

    def test_return_instance_of_list(self):
        text_preprocessor = preprocessing_text()
        word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
        actual = text_preprocessor.remove_empty_space(word_list)
        assert isinstance(actual,list)


class TestRemoveNonAscii():

    def test_no_non_ascii_character(self):
        text_preprocessor = preprocessing_text()
        word_list = ['ɳɳɳ', 'thɫɫe ', ' ', ' I', 'didn\'t', 'likɮɮɮe', ' vegetaʊʊʊbles!', 'Will ', 'I', '¥¥¥¥¥','li¥¥ke ', 'them©', '!']
        actual = text_preprocessor.remove_non_ascii(word_list)
        expected = ['', 'the ', ' ', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', '','like ', 'them', '!']
        assert actual == expected

    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.remove_non_ascii(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list = [['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.remove_non_ascii(word_list)

    def test_return_instance_of_list(self):
        text_preprocessor = preprocessing_text()
        word_list = ['ɳɳɳ', 'thɫɫe ', ' ', ' I', 'didn\'t', 'likɮɮɮe', ' vegetaʊʊʊbles!', 'Will ', 'I', '¥¥¥¥¥','li¥¥ke ', 'them©', '!']
        actual = text_preprocessor.remove_empty_space(word_list)
        assert isinstance(actual,list)


class TestTextToString():

    # def test_text_length_equals_list_lentgth(self):
    #     text_preprocessor = preprocessing_text()
    #     word_list = ['  In', 'the ', 'beginning,', ' I', 'didn\'t', 'like', ' vegetables!', 'Will ', 'I', ' ever','like ', 'them?', '!']
    #     string = text_preprocessor.word_list_to_string(word_list)
    #     actual = len(string.split(' '))
    #     print(string)
    #     print(string.split(' '))
    #     expected = len(word_list)
    #     assert actual == expected

    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.word_list_to_string(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list = [['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.word_list_to_string(word_list)

    def test_return_instance_of_string(self):
        text_preprocessor = preprocessing_text()
        word_list = ['ɳɳɳ', 'thɫɫe ', ' ', ' I', 'didn\'t', 'likɮɮɮe', ' vegetaʊʊʊbles!', 'Will ', 'I', '¥¥¥¥¥','li¥¥ke ', 'them©', '!']
        actual = text_preprocessor.word_list_to_string(word_list)
        assert isinstance(actual,str)

class TestCleanText():

    def test_return_instance_of_list(self):
        text_preprocessor = preprocessing_text()
        word_list = ['ɳɳɳ', 'thɫɫe ', ' ', ' I', 'didn\'t', 'likɮɮɮe', ' vegetaʊʊʊbles!', '  Will ', 'I', '  ', '¥¥¥¥¥','  li¥¥ke ', '  them©', '!']
        actual = text_preprocessor.clean_text(word_list)
        assert isinstance(actual,list)

    def test_that_on_non_list_argument(self):
        text_preprocessor = preprocessing_text()
        word_list = 2
        with pytest.raises(TypeError):
            text_preprocessor.clean_text(word_list)

    def test_one_dimensional(self):
        text_preprocessor = preprocessing_text()
        word_list = [['this', 'is'], ['2d', 'list']]
        with pytest.raises(ValueError):
            text_preprocessor.clean_text(word_list)

















