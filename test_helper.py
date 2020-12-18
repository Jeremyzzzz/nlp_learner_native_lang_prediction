from bs4 import BeautifulSoup
from HTMLextractor_helper import *
import sys
from zipfile import ZipFile
from feature_extraction_helper import *

ZIP_FILEPATH = "lang-8.zip"
test_text = """
I am Japanese living in Tokyo.
Sometimes I read English documents for my job, but there were few chances to write in English. 
            
I want to learn here.
"""


def load_html(filename):
    """Loaded html file to a Beatiful soup instance

    Args:
        filename (str): A raw string of filepath

    Returns:
        bs4.BeautifulSoup: A Beautiful soup instance read by lxml
    """
    with open(filename, encoding="utf-8") as f:
        html = f.read()
    return BeautifulSoup(html, "lxml")


def remove_delim(text):
    """
    Returns the clean text of the text after removing newline.

    Parameters:
    ------
    text: (str)
    the input text

    Returns:
    -------
    the clean text: (text)

    """
    return text.replace("\n", "").replace(" ", "")


def test_get_file_name():
    assert get_filename("lang-8/01.html") == "01.html"
    assert get_filename("01.html") == "01.html"
    assert get_filename("./lang-8.zip/01.html") == "01.html"
    print("get_file_name: Success!")


def test_get_text():
    assert remove_delim(
        get_text(load_html("test_samples/russia.html"))
    ) == remove_delim("Hello! My name is Kate and I want to talking with you)")
    assert remove_delim(get_text(load_html("test_samples/japan.html"))) == remove_delim(
        'Hi there,I am Japanese living in Tokyo.Sometimes I read English documents for my job, but there were few chances to write in English. I want to learn here.I like listening to / making (especially electronic) music.Recently I made a techno tune for the free online compilation album planned by a friend of mine. It will be soon released!By the way, it is really cold today. It is about time to get my "kotatsu" (Japanese table with an electric heater) ready.'
    )
    assert remove_delim(get_text(load_html("test_samples/china.html"))) == remove_delim(
        "I just known about Lang-8.But I feel very happy that I can learn English like this way.Also I think i can made friends here.I'm new.You are welcome,everyone"
    )
    print("get_text: Success!")


def test_get_L1():
    assert get_L1(load_html("test_samples/japan.html")) == "Japanese"
    assert get_L1(load_html("test_samples/russia.html")) == "Russian"
    assert get_L1(load_html("test_samples/china.html")) == "Mandarin"
    print("get_L1: Success!")


def test_corpus_reader():
    doc_counter, lang_counter, text_counter = 0, 0, 0
    my_zip = ZipFile(ZIP_FILEPATH)

    for doc, native_lang, body_text in corpus_reader(my_zip):
        doc_counter += 1
        lang_counter += 1
        text_counter += 1
    my_zip.close()
    assert doc_counter == 1545
    assert lang_counter == 1545
    assert text_counter == 1545
    print("corupus_reader: Success!")


def test_load_dataset():
    assert len(load_dataset("./test_samples/fake_file.txt")) == 7
    assert load_dataset("./test_samples/fake_file.txt")[0] == "line1"
    assert load_dataset("./test_samples/fake_file.txt")[4] == ""
    print("get_dataset: Success!")


def test_tokenize_word():
    assert tokenize_word("He loves his dog") == ["He", "loves", "his", "dog"]
    assert tokenize_word("I am so tired. I think I need a nap.") == [
        "I",
        "am",
        "so",
        "tired",
        ".",
        "I",
        "think",
        "I",
        "need",
        "a",
        "nap",
        ".",
    ]
    print("tokenize_word: Success!")


def test_if_mention_asian():
    assert if_mention_asian("I love Chinese and Japanese food") == True
    assert if_mention_asian("I visited Japan a couple of years ago") == True
    assert if_mention_asian("I visted France before") == False
    print("if_metion_asian: Success!")


def test_if_mention_european():
    assert (
        if_mention_european(
            "Once the pandemic is over, I want to travel to Spain and France"
        )
        == True
    )
    assert if_mention_european("My sister lived in Spain for four years.") == True
    assert if_mention_european("I visited China last year") == False
    print("if_mention_european: Success!")


def test_stopwords_count():
    assert stopwords_counts("He loves his dog.") == 2
    assert stopwords_counts("I love him more than myself.") == 5
    print("stopwords_count: Success!")


def test_lemmatize():
    assert lemmatize("running") == "run"
    assert lemmatize("sleeping") == "sleep"
    assert lemmatize("kitties") == "kitty"
    print("lemmatize: Success!")


def test_lemma_counts():
    assert lemma_counts("I was sleeping") == 3
    assert lemma_counts("sleep sleeping slept") == 1
    print("lemma_counts: Success!")


def test_get_length_in_words():
    assert get_length_in_words("I was worried about you") == 5
    assert get_length_in_words("I am so tired. I think I need a nap.") == 12
    assert get_length_in_words("no-word-here") == 1
    assert get_length_in_words("no-word-here!!!!") == 5
    print("get_length_in_words: Success!")


def test_get_avg_word_per_sentence():
    assert get_avg_word_per_sentence("I am Japanese living in Tokyo.") == 7
    assert (
        get_avg_word_per_sentence(
            "I am Japanese living in Tokyo. I want to learn here!"
        )
        == 13 / 2
    )
    assert get_avg_word_per_sentence(test_text) == 32 / 3
    print("get_avg_word_per_sentence: Success!")


def test_get_avg_word_length():
    assert get_avg_word_length("I am Japanese living in Tokyo.") == 4
    assert (
        get_avg_word_length("I am Japanese living in Tokyo. I want to learn here!!!")
        == 40 / 11
    )
    assert get_avg_word_length(test_text) == 116 / 28
    print("get_avg_word_length: Success!")


def test_type_token_ratio():
    assert type_token_ratio(brown.words(), 1000) == 0.417
    assert type_token_ratio(brown.words(), 100000) == 0.13082
    print("type_token_ratio: Success!")


def test_get_pos_elements():
    assert get_pos_elements("Hi there, I am Japanese living in Tokyo") == (
        4,
        1,
        1,
        0,
        1,
    )
    assert get_pos_elements("He runs very quickly") == (0, 1, 0, 2, 1)
    print("get_pos_elements: Success!")


def test_get_proposition_proba():
    assert get_preposition_proba("I got up at six as well as other day") == 2 / 10
    assert (
        get_preposition_proba(
            "I feel very happy that I can learn English like this way"
        )
        == 2 / 12
    )
    print("get_proposition_proba: Success!")


def test_count_non_open_class_tags():
    assert count_non_open_class_tags("The university is amazing.") == 2
    assert (
        count_non_open_class_tags(
            "Oh, I think I really need lots of time to improve my English now"
        )
        == 7
    )
    print("count_non_open_class_tags: Success!")


def test_count_web_words():
    assert count_web_text("This is a beauuuuuutiful night.") == 1
    assert (
        count_web_text(
            "The Fulton County Grand Jury said Friday an investigation of Atlanta's recent primary election produced 'no evidence' that any irregularities took place."
        )
        > 0
    )
    print("count_web_words: Success!")


if __name__ == "__main__":
    test_get_file_name()
    test_get_text()
    test_get_L1()
    # test_corpus_reader()
    test_load_dataset()
    test_tokenize_word()
    test_if_mention_asian()
    test_if_mention_european()
    test_stopwords_count()
    test_lemmatize()
    test_lemma_counts()
    test_get_length_in_words()
    test_get_avg_word_per_sentence()
    test_get_avg_word_length()
    test_type_token_ratio()
    test_get_pos_elements()
    test_get_proposition_proba()
    test_count_non_open_class_tags()
    test_count_web_words()
