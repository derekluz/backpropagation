import csv
import numpy as np

def lineToArray(data):
    line = data.readline()
    reader = csv.reader(line.split('\n'), delimiter=',')
    rows = []
    for row in reader:
        rows.append(row)
    gray_list = [int(x) for x in rows[0]]
    outputIndex = gray_list[0]
    gray_list[0] = 1
    array = np.asarray(gray_list)
    output = np.zeros(10)
    output[outputIndex] = 1
    return array, output, outputIndex

def sigmoid(v):
    return 1.0/(1.0 + np.exp(-v))

def sigmoidDerivate(v):
    return sigmoid(v) * (1.0 - sigmoid(v))

def initWeightMat(inputSize, outputSize):
    return np.random.rand(outputSize,inputSize+1)

def activationFunction(weightMatrix, inputArray, extendBias):
    inputSum = np.dot(weightMatrix, inputArray)
    output = [sigmoid(v) for v in inputSum]
    if extendBias:
        extendedOutputList = np.concatenate(([1], output), axis=0)
        extendedOutput = np.asarray(extendedOutputList)
    else:
        extendedOutput = np.asarray(output)
    return inputSum, extendedOutput

def updateWeightMat(weightMat, learningRate, localGradients, outPreviousLayer):
    for i in xrange(weightMat.shape[0]):
        for j in xrange(weightMat.shape[1]):
            weightMat[i,j] += learningRate * localGradients[i] * outPreviousLayer[j]
    return weightMat



def backPropagation(learningRate, targetOutput, output, hiddenWeightMat, outputWeightMat, inSumHidden, inSumOut, outHidden, inputArray):
    # Goes from output layer to hidden layer
    error = targetOutput - output
    sigDerivateInputOut = np.asarray([sigmoidDerivate(v) for v in inSumOut])
    localGradientsOut = np.multiply(sigDerivateInputOut, error)
    newOutputWeightMat = updateWeightMat(outputWeightMat, learningRate, localGradientsOut, outHidden)

    # Goes from hidden layer to input layer
    sigDerivateInputHidden = np.asarray([sigmoidDerivate(v) for v in inSumHidden])
    relativeError = np.dot(localGradientsOut, outputWeightMat[:,1:])
    localGradientsHidden = np.multiply(sigDerivateInputHidden, relativeError)
    newHiddenWeightMat = updateWeightMat(hiddenWeightMat, learningRate, localGradientsHidden, inputArray)
    return newHiddenWeightMat, newOutputWeightMat

def lossFunction(targetOutput, outputArray):
    loss = 0.0
    for k in xrange(outputArray.size):
        if k == targetOutput:
            yk = 1
        else:
            yk = 0
        loss += -yk * np.log10(outputArray[k]) - (1-yk) * np.log10(1 - outputArray[k])
    return loss


sampleNum = 5000
inputSize = 784
hiddenSize = 25
outputSize = 10
learningRate = 0.5
epocaNum = 1000

# parameters to be tested
hiddenSize_array = [25, 50, 100]
batchSize_array = [1, 10, 50, 5000]
learningRate_array = [0.5, 1, 10]

# initialize weight matrixes with random values
hiddenWeightMat = initWeightMat(inputSize, hiddenSize)
outputWeightMat = initWeightMat(hiddenSize, outputSize)

for i in xrange(epocaNum):
    data = None
    try:
        data = open("data_tp1", "r")
        loss = 0.0
        for j in xrange(sampleNum):
            # Goes from input layer to hidden layer
            inputArray, targetOutput, outputIndex = lineToArray(data)
            inSumHidden, outHidden = activationFunction(hiddenWeightMat, inputArray, True)

            # Goes from hidden layer to output layer
            inSumOut, output = activationFunction(outputWeightMat, outHidden, False)

            # Goes backwards updating the weights
            hiddenWeightMat, outputWeightMat = backPropagation(learningRate, targetOutput, output, hiddenWeightMat, outputWeightMat, inSumHidden, inSumOut, outHidden, inputArray)
            loss += lossFunction(outputIndex, output)
        print loss/sampleNum
    finally:
        if data is not None:
            data.close()
