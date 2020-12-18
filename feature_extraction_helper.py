from nltk import pos_tag, sent_tokenize, word_tokenize
from nltk.corpus import stopwords, brown, words, webtext
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
import string

LEX_ASIAN = {"Chinese", "Japanese", "Korean", "China", "Japan", "Korea"}
LEX_EUROPEAN = {"French", "France", "Spain", "Spanish"}
WORD_SET = set(webtext.words()) | set(list(string.punctuation))


def load_dataset(filepath):
    """
    Returns a list of docs from the given filepath.

    Parameters:
    filepath -- (str)

    Returns:
    -------
    list of filenames: (str)
    """

    file_list = []
    with open(filepath) as f:
        for line in f:
            file_list.append(line.strip())
    return file_list


def get_feature_dict(text):
    """
    Returns a dictionary of features extracted from text

    Parameters:
    text -- (str)

    Returns:
    -------
    dictionary of features in either boolean or numeric
    """

    feature_dict = {}

    # boolean features
    feature_dict["if_mention_european"] = if_mention_european(text)
    feature_dict["if_mention_asian"] = if_mention_asian(text)

    # statistics
    feature_dict["stopwords_counts"] = stopwords_counts(text)
    feature_dict["avg_word_count_per_sentence"] = get_avg_word_per_sentence(text)
    feature_dict["avg_word_length"] = get_avg_word_length(text)

    # POS, lemmatization, stemming
    noun_count, verb_count, adj_count, adv_count, pronoun_count = get_pos_elements(text)

    feature_dict["lemma_counts"] = lemma_counts(text)
    feature_dict["type_token_ratio"] = type_token_ratio(text, 100)

    feature_dict["noun_count"] = noun_count
    feature_dict["verb_count"] = verb_count
    feature_dict["adj_count"] = adj_count
    feature_dict["adv_count"] = adv_count
    feature_dict["pronoun_count"] = pronoun_count

    feature_dict["non_open_class_tags"] = count_non_open_class_tags(text)
    feature_dict["tag_X_count"] = count_X_tags(text)
    feature_dict["preposition_prob"] = get_preposition_proba(text)
    feature_dict["web_words_count"] = count_web_text(text)

    return feature_dict


def tokenize_word(text):
    """
    Returns the tokenized words of a given text.

    Parameters:
    text -- (str)

    Returns:
    -------
    tokenized text: (str)
    """
    return word_tokenize(text.replace("\n", ""))


def if_mention_asian(text):
    """
    Returns 1 if text mentions any of asian languages or countries, otherwise, 0.

    Parameters:
    text -- (str)

    Returns:
    -------
    the number of occurance: (int)
    """

    return 1 if any(token in LEX_ASIAN for token in tokenize_word(text)) else 0


def if_mention_european(text):
    """
    Returns 1 if text mentions any of european languages or countries, otherwise, 0.

    Parameters:
    text -- (str)

    Returns:
    -------
    the number of occurance: (int)
    """

    return 1 if any(token in LEX_EUROPEAN for token in tokenize_word(text)) else 0


def stopwords_counts(text):
    """
    Returns the number of English stopwords in the html text.

    Parameters:
    text -- (str)
    Returns:
    -------
    number of English stopwords: (str)
    """
    count = 0
    for token in tokenize_word(text):
        if token.lower() in stopwords.words("English"):
            count += 1
    return count


def lemmatize(word):
    """
    Returns the lemmatized word as noun or verb.

    Parameters:
    word -- (str)

    Returns:
    -------
    lemmatized words: (str)
    """
    lemmatizer = WordNetLemmatizer()
    lemma = lemmatizer.lemmatize(word, "v")
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, "n")
    return lemma


def lemma_counts(text):
    """
    Returns the number of lemmatizeds.

    Parameters:
    word -- (str)

    Returns:
    -------
    number of lemmatized words: (int)
    """

    lemma_dict = defaultdict(int)
    for token in tokenize_word(text):
        lemma = lemmatize(token)
        lemma_dict[lemma] += 1

    return len(lemma_dict)


def get_length_in_words(text):
    """
    Returns the length of the text in words.

    Parameters:
    ------
    text: (str)
    the input text

    Returns:
    -------
    length of tokenized text: (int)

    """
    return len(tokenize_word(text))


def get_avg_word_per_sentence(text):
    """
    Returns the average count of words per sentence of the given text (inclduing punctuation).

    Parameters:
    text -- (str)

    Returns:
    -------
    The average words per sentennce: (float)
    """
    sents = sent_tokenize(text)
    total = 0
    for sent in sents:
        total += get_length_in_words(sent)
    return total / len(sents)


def get_avg_word_length(text):
    """
    Returns the average word length of the given text.(excluding punctuation)

    Parameters:
    text: (str)
    the input text

    Returns:
    -------
    average length of tokenized text: (float)
    """
    sents = sent_tokenize(text)
    total_length = 0
    count = 0
    for sent in sents:
        words = tokenize_word(sent)
        for word in words:
            if word not in string.punctuation:
                total_length += len(word)
                count += 1
    return total_length / count


def type_token_ratio(words, num_words):
    """
    Returns the type token ration given words and number of words.

    Parameters:
    word -- (str)
    num_word -- (int)

    Returns:
    -------
    Type token ratio: (int)
    """
    types = set([word.lower() for word in words[:num_words]])
    TTR = len(types) / num_words
    return TTR


def get_pos_elements(text):
    """Return a tuple of POS counts from text

    Args:
        text (str): A raw text string

    Returns:
        A tuple of noun count, verb count, adjective count, adverb count and pronoun counts: (tuple of int)
    """
    noun_count = 0
    verb_count = 0
    adj_count = 0
    adv_count = 0
    pronoun_count = 0

    tagged = pos_tag(text.split(), tagset="universal")

    for _, pos in tagged:
        if pos == "NOUN":
            noun_count += 1
        elif pos == "VERB":
            verb_count += 1
        elif pos == "ADJ":
            adj_count += 1
        elif pos == "ADV":
            adv_count += 1
        elif pos == "PRON":
            pronoun_count += 1
    return noun_count, verb_count, adj_count, adv_count, pronoun_count


def get_preposition_proba(text):
    """
    Returns the probability of prepositions in the given text.

    Parameters:
    text: (str)

    Returns:
    -------
    The probability of prepositions: (float)
    """
    count = 0
    tagged = pos_tag(tokenize_word(text), tagset="universal")
    for _, pos in tagged:
        if pos == "ADP":
            count += 1
    return count / get_length_in_words(text)


def count_X_tags(text):
    """
    Returns the number of X tags from given text.

    Parameters:
    text: (str)

    Returns:
    -------
    The number of X tags: (int)
    """
    open_class_tags = {"ADJ", "NOUN", "ADV", "VERB"}
    count = 0
    tagged = pos_tag(tokenize_word(text), tagset="universal", lang="eng")
    for _, pos in tagged:
        if pos == "X":
            count += 1
    return count


def count_non_open_class_tags(text):
    """
    Returns the number of non open class tags from given text.

    Parameters:
    text: (str)

    Returns:
    -------
    The number of non open class tags: (int)
    """
    open_class_tags = {"ADJ", "NOUN", "ADV", "VERB"}
    count = 0
    tagged = pos_tag(tokenize_word(text), tagset="universal", lang="eng")
    for _, pos in tagged:
        if pos not in open_class_tags:
            count += 1
    return count


def count_web_text(text):
    """Count number of words appears in NLTK web corpus

    Args:
        text (str): a raw string

    Returns:
    -------
    The number of words in NLTK web corpus: (int)
    """
    for sent in sent_tokenize(text):
        token_words = tokenize_word(sent)
        count = 0
        for word in token_words:
            if word.lower() not in WORD_SET:
                count += 1
        return count