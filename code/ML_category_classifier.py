# %%
#import statements
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
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adam
import sys

task = sys.argv[1]
if task == 'train':
    train_flag = True
elif task == 'test':
    train_flag = False
    model_path = sys.argv[4]
else:
    sys.exit('Invalid Operation')

features_file = sys.argv[2]
run_no = sys.argv[3]
print('running script run no: ' + sys.argv[3])

max_val = 99999

with open(features_file) as f:
    features = [feature.strip() for feature in f]

if train_flag:
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
    labels[labels.index(i)] = i.replace('DrDoS_','')
    if i == 'UDP-lag':
        labels[labels.index(i)] = 'UDPLag'
labels = np.unique(labels)
labels = list(labels)



if train_flag:
    train_dir = '/scratch/omossad/CICDDoS2019/CSVs/01-12/'

test_dir = '/scratch/omossad/CICDDoS2019/CSVs/03-11/'
output_dir ='../cedar/output/'

# %%
#load dataset into dataframe
def read_file(filename, y_out):
    df = pd.read_csv(filename)
    df.columns = df.columns.str.strip()
    df = df[features] 
    df = df[df["Label"] != 'BENIGN']
    NewLabel = []
    for i in df["Label"]:
        i = i.replace('DrDoS_','')
        i = i.replace('DrDoS_','')
        if i == 'UDP-lag':
            i = 'UDPLag'
        NewLabel.append(labels.index(i))
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

    
nClasses = len(labels)

#### LOAD TRAIN DATA ######
if train_flag:

    new_x = pd.DataFrame()
    temp_y = []

    for f in train_files:
        print('Processing file ' + f + '\n')
        new_x = new_x.append(read_file(train_dir + f, temp_y))
        print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')

    #new_y = np.asarray(temp_y)
    new_y = to_categorical(temp_y, num_classes=nClasses)

    xTrain, xVal, yTrain, yVal = train_test_split(new_x, new_y, test_size = 0.2, random_state = 42)
    #np.savetxt('train_idx', idx1)
    #np.savetxt('test_idx', idx2)

    print('train size: ', xTrain.shape)
    print('train labels: ', yTrain.shape)
    print('Valid size: ',  xVal.shape)

#### LOAD TEST DATA #######
new_x = pd.DataFrame()
temp_y = []

for f in test_files:
    print('Processing file ' + f + '\n')
    new_x = new_x.append(read_file(test_dir + f, temp_y))
    print('Processed file ' + f + ' , total samples is ' + str(len(temp_y)) + '\n')

xTest = np.asarray(new_x)
#yTest = np.asarray(temp_y)
yTest = to_categorical(temp_y, num_classes=nClasses)

#xTest, xTemp, yTest, yTemp = train_test_split(new_x, new_y, test_size = 0.01, random_state = 42)
#np.savetxt('train_idx', idx1)
#np.savetxt('test_idx', idx2)

print('test size: ',  xTest.shape)



model = Sequential()
model.add(Dense(64, input_dim=len(features)-1, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(nClasses, activation='softmax'))
print(model.summary(90))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

num_batch = 1000
num_epochs = 10
es = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=num_epochs, mode='auto', baseline=None, restore_best_weights=True, verbose=1)

if train_flag:
    model.fit(xTrain, yTrain, batch_size=num_batch, validation_data=[xVal, yVal], epochs = num_epochs, verbose=1)
    #, callbacks=[es])
    model.save(output_dir + run_no + '/model_weights')
else:
    model = load_model(model_path)
print(model.evaluate(xTest,yTest))
model.save(output_dir + run_no + '/model_weights')


prediction = np.argmax(model.predict(xTest), axis=1)
y_test = np.argmax(yTest, axis=1)
print('\n confusion matrix: \n')
print(confusion_matrix(y_test, prediction))
np.savetxt(output_dir + run_no +'/predictions.txt', prediction)
np.savetxt(output_dir + run_no +'/ground_truth.txt', y_test)






