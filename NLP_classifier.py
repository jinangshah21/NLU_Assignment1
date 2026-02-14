import os
import math
import numpy as np
# import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# 1. LOAD DATA
def load_dataset(base_path):
    texts = []
    labels = []

    categories = ["sport", "politics"]

    for category in categories:
        folder_path = os.path.join(base_path, category)

        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(folder_path, filename)

                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read().lower()
                    texts.append(content)
                    labels.append(category)

    return texts, labels


# 2. BASIC TOKENIZATION
def tokenize(text):
    return text.split()


# 3. BUILD VOCABULARY
def build_vocabulary(texts, ngram_range=(1,1)):
    vocab = set()

    for text in texts:
        tokens = tokenize(text)

        ngrams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i:i+n])
                ngrams.append(gram)

        vocab.update(ngrams)

    vocab = sorted(list(vocab))
    word_to_index = {word: idx for idx, word in enumerate(vocab)}

    return vocab, word_to_index


# 4. BAG OF WORDS
def bag_of_words(texts, word_to_index, ngram_range=(1,1)):
    matrix = np.zeros((len(texts), len(word_to_index)))

    for doc_idx, text in enumerate(texts):
        tokens = tokenize(text)

        grams = []
        for n in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(tokens) - n + 1):
                gram = " ".join(tokens[i:i+n])
                grams.append(gram)

        counts = Counter(grams)

        for word, count in counts.items():
            if word in word_to_index:
                matrix[doc_idx][word_to_index[word]] = count

    return matrix


# 5. TF-IDF
def compute_tfidf(bow_matrix):
    tf = bow_matrix.copy()

    # Term Frequency (normalize by document length)
    for i in range(tf.shape[0]):
        doc_sum = np.sum(tf[i])
        if doc_sum != 0:
            tf[i] = tf[i] / doc_sum

    # Document Frequency
    df = np.sum(bow_matrix > 0, axis=0)

    # Inverse Document Frequency
    N = bow_matrix.shape[0]
    idf = np.log((N + 1) / (df + 1)) + 1

    # TF-IDF
    tfidf = tf * idf

    return tfidf


# 6. TRAIN & EVALUATE
def train_and_evaluate(X_train, X_test, y_train, y_test, model):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    return acc, classification_report(y_test, predictions)


# 7. PLOT FUNCTION
# def plot_accuracies(feature_name, model_names, accuracies):

#     plt.figure()
#     plt.bar(model_names, accuracies)

#     plt.title(f"Accuracy Comparison - {feature_name}")
#     plt.xlabel("Models")
#     plt.ylabel("Accuracy")
#     plt.ylim(0, 1)

#     for i in range(len(accuracies)):
#         plt.text(i, accuracies[i] + 0.01, round(accuracies[i], 3), ha='center')

#     plt.show()


def main():

    base_path = "./"

    print("Loading dataset...")
    texts, labels = load_dataset(base_path)

    print("Total documents:", len(texts))

    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42, stratify=labels
    )

    model_names = ["Naive Bayes", "Logistic Regression", "SVM"]
    models = [
        MultinomialNB(),
        LogisticRegression(max_iter=1000),
        LinearSVC()
    ]

    # BAG OF WORDS
    print("\n########## BAG OF WORDS ##########")

    vocab, word_to_index = build_vocabulary(X_train_texts, (1,1))
    X_train_bow = bag_of_words(X_train_texts, word_to_index, (1,1))
    X_test_bow = bag_of_words(X_test_texts, word_to_index, (1,1))

    bow_acc = []

    for model, name in zip(models, model_names):
        acc, report = train_and_evaluate(X_train_bow, X_test_bow, y_train, y_test, model)
        bow_acc.append(acc)

        print("\nModel:", name)
        print("Accuracy:", round(acc, 4))
        print(report)


    # TF-IDF
    print("\n########## TF-IDF ##########")

    vocab, word_to_index = build_vocabulary(X_train_texts, (1,1))
    X_train_bow = bag_of_words(X_train_texts, word_to_index, (1,1))
    X_test_bow = bag_of_words(X_test_texts, word_to_index, (1,1))

    X_train_tfidf = compute_tfidf(X_train_bow)
    X_test_tfidf = compute_tfidf(X_test_bow)

    tfidf_acc = []

    for model, name in zip(models, model_names):
        acc, report = train_and_evaluate(X_train_tfidf, X_test_tfidf, y_train, y_test, model)
        tfidf_acc.append(acc)

        print("\nModel:", name)
        print("Accuracy:", round(acc, 4))
        print(report)


    # N-GRAMS (1-2)
    print("\n########## N-GRAMS (1-2) ##########")

    vocab, word_to_index = build_vocabulary(X_train_texts, (1,2))
    X_train_ngram = bag_of_words(X_train_texts, word_to_index, (1,2))
    X_test_ngram = bag_of_words(X_test_texts, word_to_index, (1,2))

    ngram_acc = []

    for model, name in zip(models, model_names):
        acc, report = train_and_evaluate(X_train_ngram, X_test_ngram, y_train, y_test, model)
        ngram_acc.append(acc)

        print("\nModel:", name)
        print("Accuracy:", round(acc, 4))
        print(report)


if __name__ == "__main__":
    main()
