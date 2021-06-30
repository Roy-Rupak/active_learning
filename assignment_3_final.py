import os
import numpy as np
from scipy.io import loadmat
from sklearn import datasets
from sklearn import linear_model
import matplotlib.pyplot as plt

data_MMI = os.getcwd() + "/data" + "/MMI/"
data_MR = os.getcwd() + "/data" + "/MindReading/"
print("path is")
print(data_MR)
list_of_datasets = [data_MMI, data_MR]
random_accuracy_values = []
uncertain_accuracy_values = []
list_of_numbers = ["1", "2", "3"]
i = 0
for x in list_of_datasets:
    data = x
    random_accuracy_values = []
    uncertain_accuracy_values.clear()
    for y in list_of_numbers:
        if i == 0:
            trainingLabel_1 = loadmat(x + "trainingLabels_" + y, appendmat=True)
            traningMat_1 = loadmat(x + "trainingMatrix_" + y, appendmat=True)


        else:
            trainingLabel_1 = loadmat(x + "trainingLabels_MindReading_" + y, appendmat=True)
            traningMat_1 = loadmat(x + "trainingMatrix_MindReading" + y, appendmat=True)
        trainLabel = np.array(trainingLabel_1['trainingLabels'])
        trainLabel_U = trainLabel = trainLabel.T[0]
        trainMat_U = trainMat = np.array(traningMat_1['trainingMatrix'])

        print("Train Mat shape:", trainMat.shape)
        print("Train Label shape:", trainLabel.shape)
        if i == 0:

            testingLabel_1 = loadmat(x + "testingLabels_" + y, appendmat=True)
            testingMat_1 = loadmat(x + "testingMatrix_" + y, appendmat=True)

        else:
            testingLabel_1 = loadmat(x + "testingLabels_MindReading" + y, appendmat=True)
            testingMat_1 = loadmat(x + "testingMatrix_MindReading" + y, appendmat=True)

        testLabel = np.array(testingLabel_1['testingLabels'])
        testLabel_U = testLabel = testLabel.T[0]
        testMat_U = testMat = np.array(testingMat_1['testingMatrix'])
        # print(testMat)
        print("Test Mat shape:", testMat.shape)
        print("Test Label shape:", testLabel.shape)

        # unlabeled
        if i == 0:
            unlabeled_L = loadmat(x + "unlabeledLabels_" + y, appendmat=True)
            unlabeled_M = loadmat(x+ "unlabeledMatrix_" + y, appendmat=True)


        else:


            unlabeled_L = loadmat(x + "unlabeledLabels_MindReading_" + y, appendmat=True)
            unlabeled_M = loadmat(x + "unlabeledMatrix_MindReading" + y, appendmat=True)

        unlabeled_label = np.array(unlabeled_L['unlabeledLabels'])
        unlabeled_label_U = unlabeled_label = unlabeled_label.T[0]
        unlabeled_mat_U = unlabeled_mat = np.array(unlabeled_M['unlabeledMatrix'])

        print("unlabeled label shape: ", unlabeled_label.shape)
        print("unlabeled Mat shape: ", unlabeled_mat.shape)

    i = i + 1

    N = 50
    k = 10
    temp_accuracy = []
    temp_accuracy.clear()
    for i in range(N):
        # Create linear regression object
        lrmodel = linear_model.LogisticRegression()
        lrmodel.fit(trainMat, trainLabel)
        temp_accuracy.append(lrmodel.score(testMat, testLabel))
        matrixindexes = np.random.choice(np.arange(unlabeled_mat.shape[0]), 10, replace=False)
        for j in matrixindexes:
            # randNumber = np.random.randint(unlabeled_mat.shape[0])
            trainMat = np.vstack([trainMat, unlabeled_mat[j]])
            # print("unLabeled shape::", unlabeled_mat.shape)
            trainLabel = np.append(trainLabel, unlabeled_label[j])
            # remove from unlabeled data
        unlabeled_mat = np.delete(unlabeled_mat, matrixindexes, axis=0)
        # index = unlabeled_label[randNumber]
        unlabeled_label = np.delete(unlabeled_label, matrixindexes)

    random_accuracy_values.append(temp_accuracy)
    random_accuracy_values = np.sum(random_accuracy_values, axis=0)
    random_accuracy_values = np.true_divide(random_accuracy_values, 3)

    print("Random Accuracy: ", random_accuracy_values)






