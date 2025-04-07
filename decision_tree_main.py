
# -------------------------------------------------------------------------
# AUTHOR: Prabhakara Teja Kambhammettu
# FILENAME: decision_tree.py
# SPECIFICATION: Train and test decision tree classifiers using different training sets
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: ~1.5 hours
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dataSets = ['cheat_training_1.csv', 'cheat_training_2.csv', 'cheat_training_3.csv']

def parse_income(value):
    try:
        return float(value.replace("$", "").replace(",", ""))
    except AttributeError:
        return float(value)
    except ValueError:
        if isinstance(value, str) and value.endswith("k"):
            return float(value[:-1]) * 1000
        else:
            raise

for ds in dataSets:

    X = []
    Y = []

    df = pd.read_csv(ds, sep=',', header=0)
    data_training = np.array(df.values)[:,1:]

    for row in data_training:
        refund = 1 if row[0] == "Yes" else 0
        marital_status = [0, 0, 0]
        if row[1] == "Single":
            marital_status = [1, 0, 0]
        elif row[1] == "Divorced":
            marital_status = [0, 1, 0]
        elif row[1] == "Married":
            marital_status = [0, 0, 1]
        income = parse_income(row[2])
        X.append([refund] + marital_status + [income])
        Y.append(1 if row[3] == "Yes" else 2)

    total_accuracy = 0

    print(f"Running 10 experiments for training file: {ds}")

    for i in range(10):
        clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=None)
        clf = clf.fit(X, Y)

        df_test = pd.read_csv("cheat_test.csv", sep=',', header=0)
        data_test = np.array(df_test.values)[:,1:]

        correct = 0

        for data in data_test:
            refund = 1 if data[0] == "Yes" else 0
            marital_status = [0, 0, 0]
            if data[1] == "Single":
                marital_status = [1, 0, 0]
            elif data[1] == "Divorced":
                marital_status = [0, 1, 0]
            elif data[1] == "Married":
                marital_status = [0, 0, 1]
            income = parse_income(data[2])
            features = [refund] + marital_status + [income]
            class_predicted = clf.predict([features])[0]
            actual_class = 1 if data[3] == "Yes" else 2
            if class_predicted == actual_class:
                correct += 1

        accuracy = correct / len(data_test)
        print(f"Run {i+1}: Accuracy = {accuracy:.2f}")
        total_accuracy += accuracy

    final_accuracy = total_accuracy / 10
    print(f"Final average accuracy when training on {ds}: {final_accuracy:.2f}\n")
