import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle

###################################
path = 'myData'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32,32,3)
batchSizeVal = 50
epochsVal = 1
stepsPerEpochVal = 2000
######################################
images = []
classNo = []
myList = os.listdir(path)
print("Total no of classes detected ",len(myList))
noOfClasses = len(myList)
print("importing classes")

for x in range(0,noOfClasses):
    myPicList = os.listdir(path+"/"+str(x))
    for y in myPicList:
        curImg = cv2.imread(path+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(imageDimensions[0],imageDimensions[1]))
        images.append(curImg)
        classNo.append(x)
    print(x,end=" ")
print(" ")

images = np.array(images)
classNo = np.array(classNo)

print(images.shape)

#########spliting data#########
X_train, X_test,Y_train,Y_test =train_test_split(images,classNo,test_size=testRatio)
X_train,X_validation,Y_train,Y_validation = train_test_split(X_train,Y_train,test_size=valRatio)
print(X_train.shape)
print(X_test.shape)
print(X_validation.shape)
noOfSamples=[]
for x in range(0,noOfClasses):
    noOfSamples.append(len(np.where(Y_train == x)[0]))
print(noOfSamples)

# plt.figure(figsize=(10,5))
# plt.bar(range(0,noOfClasses), noOfSamples)
# plt.title("NO of images in class")
# plt.xlabel("class id")
# plt.ylabel("Number of images")
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# img = preProcessing(X_train[30])
# img = cv2.resize(img,(300,300))
# cv2.imshow("preProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing,X_train)))
X_test = np.array(list(map(preProcessing,X_test)))
X_validation = np.array(list(map(preProcessing,X_validation)))
print(X_train.shape)
##adding depth for CNN
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],X_validation.shape[2],1)

dataGen = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,zoom_range=0.2,shear_range=0.1,
                             rotation_range=10)
dataGen.fit(X_train)

Y_train = to_categorical(Y_train,noOfClasses)
Y_test = to_categorical(Y_test,noOfClasses)
Y_validation = to_categorical(Y_validation, noOfClasses)
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')




def myModel():
    noOfFilters = 60
    sizeOfFilter1 =(5,5)
    sizeofFilter2 = (3,3)
    sizeOfPool = (2,2)
    noOfNodes = 500

    model = Sequential()
    model.add((Conv2D(noOfFilters,sizeOfFilter1,input_shape=(imageDimensions[0],imageDimensions[1],1
                                                             ),activation='relu')))
    model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add((Conv2D(noOfFilters//2, sizeofFilter2, activation='relu')))
    model.add((Conv2D(noOfFilters//2, sizeofFilter2, activation='relu')))
    model.add(MaxPooling2D(pool_size=sizeOfPool))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(noOfNodes,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(noOfClasses,activation='softmax'))

    model.compile(Adam(lr=0.001),loss='categorical_crossentropy', metrics=['accuracy'])

    return model

model = myModel()
print(model.summary())

batchSizeVal = 50
epochsVal = 1
stepsPerEpochVal = 2000

history = model.fit_generator(dataGen.flow(X_train,Y_train, batch_size=batchSizeVal),steps_per_epoch=stepsPerEpochVal,epochs=epochsVal,
                    validation_data=(X_validation,Y_validation),shuffle=1)


plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('loss')
plt.xlabel('epoch')
plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title('accuracy')
plt.xlabel('epoch')
plt.show()
score = model.evaluate(X_test,Y_test,verbose=0)
print('Test Score = ',score[0])
print('Test Accuracy = ',score[1])

pickle_out = open("A/model_trained.p","wb")
pickle.dump(model,pickle_out)
pickle_out.close()

