# %%
# import statements
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt

max_val = 99999

features_file = sys.argv[1]
with open(features_file) as f:
    features = [feature.strip() for feature in f]

with open('train_files.txt') as f:
    train_files = [filename.strip() for filename in f]

with open('test_files.txt') as f:
    test_files = [filename.strip() for filename in f]

with open('labels_train.txt') as f:
    train_labels = [label.strip() for label in f]

with open('labels_test.txt') as f:
    test_labels = [label.strip() for label in f]

labels = train_labels + test_labels
for i in labels:
    labels[labels.index(i)] = i.replace('DrDoS_', '')
    if i == 'UDP-lag':
        labels[labels.index(i)] = 'UDPLag'
labels = np.unique(labels)
labels = list(labels)

# Edit these directories with proper dataset address
train_dir = '../CICDDoS2019/01-12/'
test_dir = '../CICDDoS2019/03-11/'

# %%
# load dataset into dataframe
def read_file(filename, y_out):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[features]

    NewLabel = []
    for i in df["Label"]:
        i = i.replace('DrDoS_', '')
        if i == 'UDP-lag':
            i = 'UDPLag'
        NewLabel.append(i)
    df["Label"] = NewLabel
    try:
        print("Bengin Average Flow time: \n" + str(df.loc[df['Label'] == 'BENIGN'].mean()))
    except KeyError:
        print("No Benign")

    try:
        print("For file: " + filename + " Avg Flow time (attack): \n" + str(df.loc[df['Label'] != 'BENIGN'].mean()))
    except KeyError:
        print("No Attack")

    y = df['Label']
    return y


#### LOAD TRAIN DATA ######
new_x = pd.DataFrame()
temp_y = []
nClasses = len(labels)
temp_df = pd.DataFrame()
arr = []

for i, f in enumerate(train_files):
    print('Processing file ' + f)
    s = read_file(train_dir + f, temp_y).value_counts()
    print(s)
    arr.append(s)
    print('Processed file ' + f + '\n')

print (temp_df)

for f in test_files:
    print('Processing file ' + f)
    s = read_file(test_dir + f, temp_y).value_counts()
    arr.append(s)
    print(s)
    print('Processed file ' + f + '\n')

print (arr)
total_counts = pd.Series()
for count in arr:
    total_counts = total_counts.add(count, fill_value=0)
print (total_counts)
# Label Distribution plot of the dataset
total_counts.plot(kind='bar')
plt.rcParams.update({'font.size': 25})
plt.xlabel("Flow Type", labelpad=14)
plt.ylabel("Count", labelpad=17)
plt.savefig('labels_bar.svg')
