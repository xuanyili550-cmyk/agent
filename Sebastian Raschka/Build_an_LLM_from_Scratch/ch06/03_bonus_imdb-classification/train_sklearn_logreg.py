# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

"""IMDB 影评情感分类的「经典机器学习基线」（第6章分类微调的 bonus 实验）。

本脚本不使用任何神经网络/LLM，而是用最传统的「词袋(Bag-of-Words) + 逻辑回归」
作为对照基线，用来衡量后续 GPT 微调方案到底比传统方法强多少。
同时还跑了一个「多数类」哑分类器(DummyClassifier)作为最低参照线。
运行前需先有 train.csv / validation.csv / test.csv（由 download_prepare_dataset.py 生成）。
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier


def load_dataframes():
    """从当前目录读取训练/验证/测试三个 CSV，返回三个 pandas DataFrame。"""
    df_train = pd.read_csv("train.csv")
    df_val = pd.read_csv("validation.csv")
    df_test = pd.read_csv("test.csv")

    return df_train, df_val, df_test


def eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test):
    """在训练/验证/测试三个划分上分别预测并打印准确率(accuracy)。"""
    # Making predictions
    # 用已训练模型对三个划分分别做预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculating accuracy and balanced accuracy
    # 计算各划分的准确率（balanced accuracy 相关行是原作者注释掉的备选指标，保留原样）
    accuracy_train = accuracy_score(y_train, y_pred_train)
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    accuracy_val = accuracy_score(y_val, y_pred_val)
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    accuracy_test = accuracy_score(y_test, y_pred_test)
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # Printing the results
    print(f"Training Accuracy: {accuracy_train*100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val*100:.2f}%")
    print(f"Test Accuracy: {accuracy_test*100:.2f}%")

    # print(f"\nTraining Balanced Accuracy: {balanced_accuracy_train*100:.2f}%")
    # print(f"Validation Balanced Accuracy: {balanced_accuracy_val*100:.2f}%")
    # print(f"Test Balanced Accuracy: {balanced_accuracy_test*100:.2f}%")


if __name__ == "__main__":
    df_train, df_val, df_test = load_dataframes()

    #########################################
    # Convert text into bag-of-words model
    # 把文本转成词袋(Bag-of-Words)特征：CountVectorizer 统计每条评论中各词的出现次数
    vectorizer = CountVectorizer()
    #########################################

    # 注意：只在训练集上 fit_transform（学习词表 + 转换），验证/测试集用 transform（复用同一词表），
    # 这样能避免「数据泄漏」——测试集的词表不能参与训练阶段的特征构建
    X_train = vectorizer.fit_transform(df_train["text"])
    X_val = vectorizer.transform(df_val["text"])
    X_test = vectorizer.transform(df_test["text"])
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]  # 取出各划分的标签列

    #####################################
    # Model training and evaluation
    #####################################

    # Create a dummy classifier with the strategy to predict the most frequent class
    # 哑分类器：永远预测「训练集中最常见的类别」，作为衡量模型是否真正学到东西的最低参照线
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train, y_train)

    print("Dummy classifier:")
    eval_model(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    print("\n\nLogistic regression classifier:")
    # 逻辑回归：词袋特征上的经典线性分类器；max_iter=1000 提高上限以保证在高维稀疏特征上收敛
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    eval_model(model, X_train, y_train, X_val, y_val, X_test, y_test)