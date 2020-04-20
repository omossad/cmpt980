import pandas as pd
import numpy as np
import sys
import sklearn
import glob
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectPercentile, f_classif
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE

print(pd.__version__)
print(np.__version__)
print(sys.version)
print(sklearn.__version__)


# Loading the Dataset
# Feature names
features = ["ID", "Flow ID", "Source IP", "Source Port", "Destination IP", "Destination Port", "Protocol", "Timestamp",
             "Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
             "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min", "Fwd Packet Length Mean",
             "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min", "Bwd Packet Length Mean",
             "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean", "Flow IAT Std", "Flow IAT Max",
             "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std", "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total",
             "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max", "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags",
             "Bwd URG Flags", "Fwd Header Length", "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s",
             "Min Packet Length", "Max Packet Length", "Packet Length Mean", "Packet Length Std",
             "Packet Length Variance", "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
             "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count", "Down/Up Ratio", "Average Packet Size",
             "Avg Fwd Segment Size", "Avg Bwd Segment Size", "Fwd Header Length.1","Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
             "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk","Bwd Avg Bulk Rate","Subflow Fwd Packets",
             "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes", "Init_Win_bytes_forward" , "Init_Win_bytes_backward",
             "act_data_pkt_fwd", "min_seg_size_forward", "Active Mean", "Active Std", "Active Max", "Active Min",
             "Idle Mean", "Idle Std", "Idle Max", "Idle Min", "SimillarHTTP", "Inbound", "Label"]

# Set this path to the dataset path containing the csv files
# Using skip rows and nrows (for each csv file) we select 20% test data and 80% training data
path_train = sys.argv[1]
all_files_train = glob.glob(path_train + "*.csv")
df_from_each_file_train = (pd.read_csv(f, header=None, names=features, skiprows=1000, nrows=20000) for f in all_files_train)
df = pd.concat(df_from_each_file_train, ignore_index=True)

# Used a portion of the data set for the feature exploaroion, different test directory can be loaded
path_test = sys.argv[2]
all_files_test = glob.glob(path_test + "*.csv")
df_from_each_file_test = (pd.read_csv(f, header=None, names=features, skiprows=22000, nrows=4000) for f in all_files_test)
df_test   = pd.concat(df_from_each_file_test, ignore_index=True)


# @parham: dtype for these returned 'object' but should be treated as numeric values, change "infinity" to large number

df["Flow Bytes/s"] = df["Flow Bytes/s"].replace(to_replace="Infinity", value=100000)
df["Flow Packets/s"] = df["Flow Packets/s"].replace(to_replace="Infinity", value=50000)
df_test["Flow Bytes/s"] = df_test["Flow Bytes/s"].replace(to_replace="Infinity", value=100000)
df_test["Flow Packets/s"] = df_test["Flow Packets/s"].replace(to_replace="Infinity", value=50000)

df["Flow Packets/s"] = pd.to_numeric(df["Flow Packets/s"])
df["Flow Bytes/s"] = pd.to_numeric(df["Flow Bytes/s"])
df_test["Flow Packets/s"] = pd.to_numeric(df["Flow Packets/s"])
df_test["Flow Bytes/s"] = pd.to_numeric(df["Flow Bytes/s"])

# @parham: deleted timestamp feature, different model training for timeseries!
# Flow ID, Source IP and Destination IP give us no information about attack/benign this can be later used for revewing attack flows

df.drop('Timestamp', axis=1, inplace=True)
df.drop('Flow ID', axis=1, inplace=True)
#df.drop('Protocol', axis=1, inplace=True)
df.drop('Source IP', axis=1, inplace=True)
df.drop('Destination IP', axis=1, inplace=True)
df.drop('SimillarHTTP', axis=1, inplace=True)
df.drop('ID', axis=1, inplace=True)
#df.drop('Source Port', axis=1, inplace=True)
#df.drop('Destination Port', axis=1, inplace=True)

df_test.drop('Timestamp', axis=1, inplace=True)
df_test.drop('Flow ID', axis=1, inplace=True)
#df_test.drop('Protocol', axis=1, inplace=True)
df_test.drop('Source IP', axis=1, inplace=True)
df_test.drop('Destination IP', axis=1, inplace=True)
df_test.drop('SimillarHTTP', axis=1, inplace=True)
df_test.drop('ID', axis=1, inplace=True)
#df_test.drop('Source Port', axis=1, inplace=True)
#df_test.drop('Destination Port', axis=1, inplace=True)

# Check the dimension of the picked data
print('Number of features:', len(features))
print('Shape of the Training :',df.shape)
print('Shape of the Test : ', df_test.shape)

#Check Label distributions (Should be evenly distributed in ideal case)
print('Label distribution Training set:')
print(df['Label'].value_counts())
print()
print()
print('Label distribution Test set:')
print(df_test['Label'].value_counts())

#Check wich features have categorical values in test set and training set
print('\n In training set:')
for feature in df.columns:
    if df[feature].dtypes == 'object' :
        print ("Featrue " + feature + " has " + str(len(df[feature].unique())) + " different categories")
print('\n In test set: ')
for feature in df_test.columns:
    if df_test[feature].dtypes == 'object' :
        feature_count = len(df_test[feature].unique())
        print("Feature " + feature + " has " + str(len(df_test[feature].unique())) + " different categories")

labels=df['Label']
labels_test=df_test['Label']

# Replace string labels with values for each class of attack
labels_to_replace = {'BENIGN' : 0, 'TFTP' : 1 ,'Syn': 2, 'DrDoS_UDP': 3, 'DrDoS_DNS': 4, 'Portmap': 5, 'DrDoS_LDAP': 6,
                 'DrDoS_SSDP': 7, 'UDP-lag': 8, 'DrDoS_SNMP': 9, 'DrDoS_NTP': 10, 'DrDoS_NetBIOS': 11,
                 'DrDoS_MSSQL': 12, 'WebDDoS' : 13}

newlabels=labels.replace(labels_to_replace)
newlabels_test=labels_test.replace(labels_to_replace)


df['Label'] = newlabels
df_test['Label'] = newlabels_test

x_train = df.drop('Label',1)
y_train = df.Label
# test set
x_test = df_test.drop('Label',1)
y_test = df_test.Label

column_names = list(x_train)
column_names_test = list(x_test)

x_train.fillna(x_train.mean(), inplace=True)
x_test.fillna(x_test.mean(), inplace=True)
# Feature scaling to make sure our basic results for feature selection are correct
scaler_train = preprocessing.StandardScaler().fit(x_train)
x_train = scaler_train.transform(x_train)
scaler_test = preprocessing.StandardScaler().fit(x_test)
x_test = scaler_test.transform(x_test)

print(x_train.std(axis=0))

# No NaN value in train and test!
where_are_NaNs = np.isnan(x_train)
x_train[where_are_NaNs] = 0
where_are_NaNs = np.isnan(x_test)
x_test[where_are_NaNs] = 0

# Decision tree classifier used as input of the recursive feature elimination
feature_selection_model=DecisionTreeClassifier(random_state=0)
feature_selection_model.fit(x_train, y_train.astype('int'))
feature_selector = RFECV(estimator=feature_selection_model, step=1, cv=10, scoring='accuracy')
feature_selector.fit(x_test, y_test)

# Plot number of features vs. number of correct classifications for determining number of features
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (number of correct classifications)")
plt.title('RFECV DoS')
plt.plot(range(1, len(feature_selector.grid_scores_) + 1), feature_selector.grid_scores_)
plt.savefig('RFECV_DDoS.png')

#After observing the number of critical features now we select 28 features
decision_tree = DecisionTreeClassifier(random_state=0)
rfe = RFE(estimator=decision_tree, n_features_to_select=28, step=1)
rfe.fit(x_train, y_train.astype('int'))
true=rfe.support_
rfe_feature_index = [i for i, x in enumerate(true) if x]
rfe_feature_names = list(column_names[i] for i in rfe_feature_index)
print('Selected features: ', rfe_feature_names)

