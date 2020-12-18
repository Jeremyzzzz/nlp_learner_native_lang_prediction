from HTMLextractor_helper import corpus_reader
from feature_extraction_helper import load_dataset, get_feature_dict
from zipfile import ZipFile
from tqdm import tqdm
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.tree import export_graphviz
import re
from nltk.corpus import stopwords, brown, words, webtext
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree

ZIP_FILEPATH = "lang-8.zip"
TRAIN_FILEPATH = "train.txt"
DEV_FILEPATH = "dev.txt"
TEST_FILEPATH = "test.txt"

ASIAN_LANG = ["Korean", "Japanese", "Mandarin"]


def split_dataset_and_extract_features(train_data, dev_data, test_data, filepath):
    """Split the dataset from the given filepath and extract featrues for trainining data, development data and test data

    Args:
        train_set (list): List of filenames in training set
        dev_set (list): List of filenames in dev set
        test_set (list): List of filenames in test set
        filepath (str): a path to the zip file

    Returns:
        Tuple of list: A tuple of list of feature dicts which extracted from zip file for each set
    """
    X_train, y_train = [], []
    X_dev, y_dev = [], []
    X_test, y_test = [], []

    my_zip = ZipFile(filepath)
    corpus_generator = corpus_reader(my_zip)
    ALL_TEXT = ""
    for doc, native_lang, body_text in tqdm(corpus_generator):
        if native_lang == "Russian":
            continue
        feature_dict = get_feature_dict(body_text)

        if doc in train_set:
            X_train.append(feature_dict)
            if native_lang in ASIAN_LANG:
                y_train.append("Asian")
            else:
                y_train.append("European")

        if doc in dev_set:
            X_dev.append(feature_dict)
            if native_lang in ASIAN_LANG:
                y_dev.append("Asian")
            else:
                y_dev.append("European")

        if doc in test_set:
            X_test.append(feature_dict)
            if native_lang in ASIAN_LANG:
                y_test.append("Asian")
            else:
                y_test.append("European")

    my_zip.close()
    return X_train, y_train, X_dev, y_dev, X_test, y_test


def feature_ablation(model, feature_lst, X_train, y_train, X_dev, y_dev):
    """Feature selection algorithm from given features

    Args:
        model (sklearn model): A sklearn model used to evaluate feature selection
        feature_lst (list): A list of d feature names
        X_train (numpy array): X_train with shape n * d
        y_train (numpy array): y_train with shape n * 1
        X_dev (numpy array): X_dev with shape m * d
        y_dev (numpy array): y_dev with shape m * 1
    """
    features = feature_lst[:]
    model.fit(X_train, y_train)
    best_score = model.score(X_dev, y_dev)
    print(f"--- The preliminary score is {best_score} ---")

    for i in range(len(features)):
        current_best_score = 0
        remove_feature = None
        remove_idx = None

        # try deleting each feature at a time and compare the validation score
        for idx, feature in enumerate(features):
            X_train_enc_ablation = np.delete(X_train, idx, 1)
            X_dev_enc_ablation = np.delete(X_dev, idx, 1)

            model.fit(X_train_enc_ablation, y_train)
            score = model.score(X_dev_enc_ablation, y_dev)
            print(f"Try removing feature: {feature}...")
            print(f"        ....reaching validation score: {score}")
            # track the best validation score when removing a feature
            if score > current_best_score:
                current_best_score = score
                remove_feature = feature
                remove_idx = idx

        # if removing any of the feature results in lower score, we should stop
        # because all of them are useful
        if current_best_score < best_score:
            break
        best_score = current_best_score
        print("***************************************")
        print(f"Removed feature: {remove_feature}")
        print(f"The best score so far is {best_score}")
        print("***************************************")

        X_train = np.delete(X_train, remove_idx, 1)
        X_dev = np.delete(X_dev, remove_idx, 1)
        features.pop(remove_idx)
    print("--- Stop removing features ---")
    print(f"The best score we can reach after feature ablation is {best_score}")
    return features


train_set = load_dataset(TRAIN_FILEPATH)
dev_set = load_dataset(DEV_FILEPATH)
test_set = load_dataset(TEST_FILEPATH)

# transform the feature_dict to numpy array
X_train, y_train, X_dev, y_dev, X_test, y_test = split_dataset_and_extract_features(
    train_set, dev_set, test_set, ZIP_FILEPATH
)
vectorizer = DictVectorizer()
X_train_enc = vectorizer.fit_transform(X_train).toarray()
X_dev_enc = vectorizer.fit_transform(X_dev).toarray()
X_test_enc = vectorizer.fit_transform(X_test).toarray()

features = [k for k, v in sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1])]

# init a decision tree model and calculate the preliminary score
model_dt = DecisionTreeClassifier(max_depth=3, random_state=123)
model_dt.fit(X_train_enc, y_train)
print(f"The preliminary score is {model_dt.score(X_dev_enc, y_dev)}.")

# feature ablation
feature_ablation(model_dt, features, X_train_enc, y_train, X_dev_enc, y_dev)

v = vectorizer.vocabulary_
selected_feature_idx = [
    v["avg_word_length"],
    v["if_mention_european"],
    v["stopwords_counts"],
    v["tag_X_count"],
    v["web_words_count"],
]
print(f"The mapping between the feature and index in DictVectorizer:{v}")
print(f"Selected index of features are {selected_feature_idx}")

X_train_new = X_train_enc[:, selected_feature_idx]
X_test_new = X_test_enc[:, selected_feature_idx]

# train decision tree with reduced feature set
model_dt.fit(X_train_new, y_train)
model_dt.score(X_test_new, y_test)

# report
model_dt_reduced = DecisionTreeClassifier(max_depth=3, random_state=123)
model_dt_reduced.fit(X_train_new, y_train)

reduced_features = [
    "avg_word_length",
    "if_mention_european",
    "stopwords_counts",
    "tag_X_count",
    "web_words_count",
]

# Visualize decision tree
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(model_dt_reduced, feature_names=reduced_features, fontsize=10)
plt.show()

# correlation between features and target
df = pd.DataFrame(data=X_train_enc, columns=features)
cor = df.corr()
plt.figure(figsize=(15, 15))
sns.set(font_scale=0.8)
sns.heatmap(cor, annot=True, cmap=plt.cm.Blues)