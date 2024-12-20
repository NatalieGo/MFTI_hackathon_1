#!/usr/bin/python

# This is a simple echo bot using the decorator mechanism.
# It echoes any incoming text messages.

import telebot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import BaggingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, recall_score

# bot token
# Токен сохранён в отдельном файле config.py. Сохранён на сервере.
# Не включен в рeпозиторий
from config import *
API_TOKEN = API_TOKEN_CONFIG


def ml_model(sentence):
    dataset_path = 'C:/Study/МФТИ/Хакатоны/Хакатон_декабрь_2024/Проект/dataset/dataset_clean_ru.csv'
    df = pd.read_csv(dataset_path, sep=';')
    X = df['post']
    y = df['category']

    vectorizer = TfidfVectorizer(max_features=50000)
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    repot_MultinomialNB = classification_report(y_test, y_pred)
    transformed_sentence = vectorizer.transform(sentence)
    # Проверка результата
    y_pred = clf.predict(transformed_sentence)
    if (y_pred == 0):
        result = "Суицидальные коннотации не найдены"
    else:
        result = "Найдены суицидальные коннотации"
    return result


bot = telebot.TeleBot(API_TOKEN)


# Handle '/start' and '/help'
@bot.message_handler(commands=['help', 'start'])
def send_welcome(message):
    bot.reply_to(message, """\
Привет, я бот, который обучен на большой выборке данных определять в постах суицидальные коннотации.
Отправь сюда интересующий тебя пост и я его проверю\
""")


# Handle all other messages with content_type 'text' (content_types defaults to ['text'])
@bot.message_handler(content_types=["text"])
def answer_message(message):
    sentence = [message.text]
    bot_answer = ml_model(sentence)
    bot.reply_to(message, bot_answer)

if __name__ == '__main__':
    bot.infinity_polling()