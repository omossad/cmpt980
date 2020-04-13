# %%
#import statements
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import sys
run_no = sys.argv[2]
print('running script run no: ' + sys.argv[2])


max_val = 99999

features_file = sys.argv[1]
with open(features_file) as f:
    features = [feature.strip() for feature in f]

with open('train_files.txt') as f:
    train_files = [filename.strip() for filename in f]

with open('test_files.txt') as f:
    test_files = [filename.strip() for filename in f]

train_dir = '/scratch/omossad/CICDDoS2019/CSVs/01-12/'
test_dir = '/scratch/omossad/CICDDoS2019/CSVs/03-11/'


# %%
#load dataset into dataframe
def read_file(filename, y_out):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[features] 
    NewLabel = []
    for i in df["Label"]:
        if i =="BENIGN":
            NewLabel.append(1)
        else:
            NewLabel.append(0)
    df["Label"]=NewLabel
    y = df['Label'].values
    y_out = y_out.extend(y)
    del df['Label']

    df = df.replace('Infinity', max_val)
    x = df.values
    scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
    scaled_df = scaler.fit_transform(x)
    #scaled_df=(df-df.mean())/df.std()
    x = pd.DataFrame(scaled_df)
    return x
    

#### LOAD TRAIN DATA ######


new_x = pd.DataFrame()
temp_y = []
nClasses = 2

for f in train_files:
    print('Processing file ' + f + '\n')
    new_x = new_x.append(read_file(train_dir + f, temp_y))
    print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')

xTrain = np.asarray(new_x)
yTrain = np.asarray(temp_y)


print('train size: ', xTrain.shape)
print('train labels: ', yTrain.shape)

#### LOAD TEST DATA #######
new_x = pd.DataFrame()
temp_y = []

for f in test_files:
    print('Processing file ' + f + '\n')
    new_x = new_x.append(read_file(test_dir + f, temp_y))
    print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')

xTest = np.asarray(new_x)
yTest = np.asarray(temp_y)


print('test size: ',  xTest.shape)



# %%
#begin individual classifier training and evaluation
#AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=50, learning_rate=1)
ada.fit(xTrain,yTrain)
yPred = ada.predict(xTest)
print("AdaBoostClassifier Performance Metrics")
print("Accuracy Score: ", accuracy_score(yTest,yPred))
print("Precision Score: ", precision_score(yTest,yPred))
print("Recall Score: ", recall_score(yTest,yPred))
print("F1 Score: ", f1_score(yTest,yPred))
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
fpr = (fp / (fp + tn)) * 100
print("False Positive Rate: " +str(fpr))
print("AdaBoostClassifier Confusion Matrix:")
print("True Positives: " +str(tp))
print("False Positives: " +str(fp))
print("True Negatives: " +str(tn))
print("False Negatives: " +str(fn))

# %%
#CART
cart = DecisionTreeClassifier()
cart.fit(xTrain,yTrain)
yPred = cart.predict(xTest)
print("DecisionTreeClassifier Performance Metrics")
print("Accuracy Score: ", accuracy_score(yTest,yPred))
print("Precision Score: ", precision_score(yTest,yPred))
print("Recall Score: ", recall_score(yTest,yPred))
print("F1 Score: ", f1_score(yTest,yPred))
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
fpr = (fp / (fp + tn)) * 100
print("False Positive Rate: " +str(fpr))
print("DecisionTreeClassifier Confusion Matrix:")
print("True Positives: " +str(tp))
print("False Positives: " +str(fp))
print("True Negatives: " +str(tn))
print("False Negatives: " +str(fn))

# %%
#Naive Bayes
nb = GaussianNB()
nb.fit(xTrain, yTrain)
yPred = nb.predict(xTest)
print("Naive Bayes Performance Metrics")
print("Accuracy Score: ", accuracy_score(yTest,yPred))
print("Precision Score: ", precision_score(yTest,yPred))
print("Recall Score: ", recall_score(yTest,yPred))
print("F1 Score: ", f1_score(yTest,yPred))
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
fpr = (fp / (fp + tn)) * 100
print("False Positive Rate: " +str(fpr))
print("Naive Bayes Confusion Matrix:")
print("True Positives: " +str(tp))
print("False Positives: " +str(fp))
print("True Negatives: " +str(tn))
print("False Negatives: " +str(fn))

# %%
#KNeighborsClassifier
'''
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(xTrain, yTrain) 
yPred = neigh.predict(xTest)
print("KNN Performance Metrics")
print("Accuracy Score: ", accuracy_score(yTest,yPred))
print("Precision Score: ", precision_score(yTest,yPred))
print("Recall Score: ", recall_score(yTest,yPred))
print("F1 Score: ", f1_score(yTest,yPred))
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
fpr = (fp / (fp + tn)) * 100
print("False Positive Rate: " +str(fpr))
print("KNN Confusion Matrix:")
print("True Positives: " +str(tp))
print("False Positives: " +str(fp))
print("True Negatives: " +str(tn))
print("False Negatives: " +str(fn))
'''
# %%
#RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=None)
rf.fit(xTrain, yTrain)
yPred = rf.predict(xTest)
print("Random Forest Performance Metrics")
print("Accuracy Score: ", accuracy_score(yTest,yPred))
print("Precision Score: ", precision_score(yTest,yPred))
print("Recall Score: ", recall_score(yTest,yPred))
print("F1 Score: ", f1_score(yTest,yPred))
tn, fp, fn, tp = confusion_matrix(yTest, yPred).ravel()
fpr = (fp / (fp + tn)) * 100
print("False Positive Rate: " +str(fpr))
print("Random Forest Confusion Matrix:")
print("True Positives: " +str(tp))
print("False Positives: " +str(fp))
print("True Negatives: " +str(tn))
print("False Negatives: " +str(fn))





