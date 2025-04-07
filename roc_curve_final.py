
# -------------------------------------------------------------------------
# AUTHOR: Prabhakara Teja Kambhammettu
# FILENAME: roc_curve.py
# SPECIFICATION: Generates and plots ROC curve from decision tree classifier
# FOR: CS 5990 (Advanced Data Mining) - Assignment #3
# TIME SPENT: ~1 hour
# -----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
import numpy as np
import pandas as pd

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

# read the dataset cheat_data.csv and prepare the data_training numpy array
df = pd.read_csv("cheat_data.csv")
data_training = np.array(df.values)

# transform features to numbers and one-hot encode 'Marital Status'
X = []
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

# transform classes to numbers: Yes = 1, No = 0
y = [1 if label == "Yes" else 0 for label in data_training[:, 3]]

# split into train/test sets using 30% for test
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.3, random_state=42)

# generate random predictions for a no-skill model
ns_probs = [0 for _ in range(len(testy))]

# fit a decision tree model by using entropy with max depth = 2
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2)
clf = clf.fit(trainX, trainy)

# predict probabilities for all test samples
dt_probs = clf.predict_proba(testX)
dt_probs = dt_probs[:, 1]  # keep probabilities for positive class

# calculate scores by using both classifiers (no skill and decision tree)
ns_auc = roc_auc_score(testy, ns_probs)
dt_auc = roc_auc_score(testy, dt_probs)

# summarize scores
print('No Skill: ROC AUC=%.3f' % (ns_auc))
print('Decision Tree: ROC AUC=%.3f' % (dt_auc))

# calculate roc curves
ns_fpr, ns_tpr, _ = roc_curve(testy, ns_probs)
dt_fpr, dt_tpr, _ = roc_curve(testy, dt_probs)

# plot the roc curve for the model
pyplot.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
pyplot.plot(dt_fpr, dt_tpr, marker='.', label='Decision Tree')
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
pyplot.legend()
pyplot.show()
