import matplotlib.pyplot as plt
import numpy as np
import json as js
from pandas import read_csv
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

def main():
    batchSize = 100
    numEpochs = 1000 
    alpha = 0.01 # The learning rate
    numHiddens = [16 ,32] # 1st hidden layer is 16 and the 2nd hidden layer is 32
    classes = [0, 1, 2, 3, 4] # Samples worked in this prokect are from 1 to 4

    # Comment one of the 2 lines below depending on if you are going to train or load the model
    model, history, xTest, yTest = trainModel(batchSize,numEpochs, alpha, numHiddens, classes)
    # model, history, xTest, yTest = loadModel("mnistModel.keras", "trainHistory.json", "xTest.npy", "yTest.npy")

    printConfusionMatrix(model, xTest, yTest)
    displayMetrics(model, history, xTest, yTest)

# Helper functions

def loadModel(modelFile, historyFile, xTestFile, yTestFile):
    model = keras.models.load_model(modelFile)
    keras.models.load_model(modelFile)
    with open(historyFile, "r") as f:
        history = js.load(f)
    xTest = np.load(xTestFile)
    yTest = np.load(yTestFile)
    return model, history, xTest, yTest

def displayMetrics(model, history, xTest, yTest):
    _ ,testAccuracy = model.evaluate(xTest, yTest, verbose=2) # _ is an uncessary variable that won't be used
    testError = 1 - testAccuracy
    print("Test Accuracy:", testAccuracy)
    print("Test Error:", testError)

    # Create window to plot both subplots
    plt.figure(figsize=(12, 5))

    # Create plot for training and validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label="Training Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Find neccsary data for training and validation error plot
    trainingError = calculateErrorsInHistory(history, "accuracy")
    validationError = calculateErrorsInHistory(history, "val_accuracy")

    # Create plot for training and validation error
    plt.subplot(1, 2, 2)
    plt.plot(trainingError, label="Training Error")
    plt.plot(validationError, label="Validation Error")
    plt.title("Training and Validation Error")
    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.legend()

    plt.show()

def calculateErrorsInHistory(history, historyKey):
    errors = []
    for epoch in range(len(history[historyKey])):
        accuracy = history[historyKey][epoch]
        errors.append(1 - accuracy) # 1 - accuracy is the error value
    return errors 

def printConfusionMatrix(model, xTest, yTest):
    print()
    print("Confusion Matrix")
    predictions = model.predict(xTest)
    yPredictions = np.argmax(predictions, axis=1)
    print(confusion_matrix(yTest, yPredictions))
    print()

def trainModel(batchSize,numEpochs, alpha, numHiddens, classes):
    trainDf, testDf = loadData(batchSize, numEpochs, alpha, numHiddens) 
    trainDf, testDf = filterData(trainDf, testDf, classes)
    xTrain, xTest, yTrain, yTest = convertDataFrameToNumPyArray(trainDf, testDf)
    # Seperate training and validation
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.1, random_state=21) # test_size is 10% percent training of the data set
    model = buildAndCompileNeuralNetwork(numHiddens, alpha, classes)
    
    history = model.fit(xTrain, yTrain, epochs=numEpochs, batch_size=batchSize, validation_data=(xVal, yVal), verbose=2) 

    with open("trainHistory.json", "w") as f:
        js.dump(history.history, f)
    model.save("mnistModel.keras")
    np.save("xTest.npy", xTest)
    np.save("yTest.npy", yTest) 

    return model, history.history, xTest, yTest 

def loadData(batchSize, numEpochs, alpha, numHiddens):
    trainDf = read_csv("mnist_train.csv")
    testDf = read_csv("mnist_test.csv")
    return trainDf, testDf

def filterData(trainDf, testDf, classes):
    trainDf = trainDf[trainDf["label"].isin(classes)]
    testDf = testDf[testDf["label"].isin(classes)]
    return trainDf, testDf

def convertDataFrameToNumPyArray(trainDf, testDf):
    #  Remove the label column and converts DataFrame into a NumPy array
    xTrain = trainDf.drop("label", axis=1).values
    xTest = testDf.drop("label", axis=1).values

    # Converts DataFrame into a NumPy array
    yTrain = trainDf["label"].values
    yTest = testDf["label"].values 

#   Normalize the data
    xTrain = xTrain / 255.0
    xTest = xTest / 255.0

    return xTrain, xTest, yTrain, yTest

def buildAndCompileNeuralNetwork(numHiddens, alpha, classes):
    model = keras.models.Sequential([
        keras.layers.Dense(numHiddens[0], activation="relu", input_shape=(28*28,)), # Image size is 28 by 28
        keras.layers.Dense(numHiddens[1], activation="relu"),
        keras.layers.Dense(len(classes), activation="softmax") 
    ])

    model.compile(optimizer=keras.optimizers.SGD(learning_rate=alpha),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

if __name__ == "__main__":
    main()
