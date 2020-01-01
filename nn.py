import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential,Model
from keras.layers import Input,concatenate,Conv2D,Dense,BatchNormalization,MaxPooling2D,Flatten,Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score,StratifiedKFold
import pandas as pd
import pickle
import numpy as np

####################################
data_file_china = pd.read_csv("data_china.csv")
data_file_monty = pd.read_csv("data.csv")


file = open('img_array_china.pkl', 'rb')
image_array_china = pickle.load(file)
file.close()


file = open('img_array_complete.pkl', 'rb')
image_array_monty = pickle.load(file)
file.close()

monty = np.load("monty.npy")
china = np.load("china.npy")

final_img_array = np.concatenate((image_array_china,image_array_monty),axis=0)
final_labels_array = np.concatenate((china,monty),axis=0)


############################################

for i in range(len(final_labels_array)):
    if final_labels_array[i]!="normal":
        final_labels_array[i] = "abnormal"
        
           #####################

labels_china = labels
for i in range(len(labels_china)):
    if(labels_china[i]!="normal"):
        labels_china[i] = "abnormal"

        
############################################

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cond_rep = le.fit_transform(final_labels_array)
final_labels_array = cond_rep

            #####################

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
cond_rep = le.fit_transform(np.array(labels_china))
labels_china = cond_rep


#################################################

def create_baseline():
    model = Sequential()

   
    
    model.add(Conv2D(16,(3,3),input_shape=(512,512,3),padding="same"))
    #model.add(MaxPooling2D((3,3)))
    #model.add(BatchNormalization())
    
   
    
    model.add(Conv2D(8,(3,3),padding="same"))
    model.add(MaxPooling2D((3,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))    

    
    model.add(Conv2D(8,(3,3),padding="same"))
    model.add(MaxPooling2D((3,3)))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))
    
    
    model.add(Flatten())
    #model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    
#####################################################

labels = data_file["condition"]

########################################################

'''Model 2'''
def create_Model():
    m_input = Input(shape=(224,224,3),name='main_input')
    
    x1 = Conv2D(16,(3,3),padding='same')(m_input)
    x2 = Conv2D(8,(3,3),padding='same')(m_input)
    x3 = Conv2D(4,(8,8),padding='same')(m_input)
    inter_block = concatenate([x1,x2,x3])
    x_ans = Conv2D(8,(3,3),padding='same',name="conv_after1")(inter_block)
    x_ans = Dropout(0.4)(x_ans)
    x_ans = Dense(8,activation='relu')(x_ans)
    x_ans = Flatten()(x_ans)
    op = Dense(1,activation='sigmoid')(x_ans)
    
    model = Model(inputs=[m_input],outputs=[op])
    return model
    
model = create_Model()    

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(image_array,labels_china,validation_split=0.2,epochs=5)

###################### Baseline Model used ########################################

model = create_baseline()
model.fit(final_img_array,final_labels_array,epochs=10,validation_split=0.33)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(half_image_array,labels,validation_split=0.2,epochs=5)




#####################################################
estimator = KerasClassifier(build_fn=create_baseline, epochs=10, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=5,shuffle=True)
results = cross_val_score(estimator,image_array,labelsS,cv=kfold)
print("Baseline Accuracy: ",results.mean()*100)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(image_array,labels_china)


model.fit(X_train,y_train,epochs=10)


y_pred = model.predict(X_test)

from sklearn.metrics import roc_curve,auc
fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test, y_pred)
auc_keras = auc(fpr_keras,tpr_keras)



####Half image array###############################

half_image_array = np.zeros(shape=(138,112,448,3))
for i in range(len(image_array)):
    
    half_image_array[i] = image_array[i][:len(image_array[i])//2]
    

#########################################################