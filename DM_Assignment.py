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
list_of_datasets=[data_MMI,data_MR]
random_accuracy_values = []
uncertain_accuracy_values = []
list_of_numbers=["1","2","3"]
i=0
for x in list_of_datasets:
    data=x
    random_accuracy_values=[]
    uncertain_accuracy_values.clear()
    for y in list_of_numbers:
        if i==0:

            trainingLabels = loadmat(x + "trainingLabels_" + y, appendmat=True)
            traningMatrix = loadmat(x + "trainingMatrix_" + y, appendmat=True)
        else:
            trainingLabels = loadmat(x + "trainingLabels_MindReading_" + y, appendmat=True)
            traningMatrix = loadmat(x + "trainingMatrix_MindReading" + y, appendmat=True)
        trainLabel = np.array(trainingLabels['trainingLabels'])
        #print(trainLabel)
        trainLabel_uncertain = trainLabel.T[0]
        trainLabel_random = trainLabel.T[0]
        trainMat_uncertain = np.array(traningMatrix['trainingMatrix'])
        trainMat_random = np.array(traningMatrix['trainingMatrix'])

        print("Train Mat shape:", trainMat_random.shape)
        print("Train Label shape:", trainLabel.shape)
        if i == 0:
            #testingLabel_1 = loadmat(x + "testingLabels_" + y, appendmat=True)
            testingLabels = loadmat(x + "testingLabels_" + y, appendmat=True)
            testingMatrix = loadmat(x + "testingMatrix_" + y, appendmat=True)
        else:
            testingLabels = loadmat(x + "testingLabels_MindReading" + y, appendmat=True)
            testingMatrix = loadmat(x + "testingMatrix_MindReading" + y, appendmat=True)


        testLabel = np.array(testingLabels['testingLabels'])
        testLabel_uncertain = testLabel.T[0]
        testLabel_random = testLabel.T[0]
        testMat_uncertain =  np.array(testingMatrix['testingMatrix'])
        testMat_random =np.array(testingMatrix['testingMatrix'])
        print(testMat_random)
        print("Test Mat shape:", testMat_random.shape)
        print("Test Label shape:", testLabel.shape)

        # unlabeled
        if i == 0:
            unlabeled_labels = loadmat(x + "unlabeledLabels_" + y, appendmat=True)
            unlabeled_matrix = loadmat(x + "unlabeledMatrix_" + y, appendmat=True)

        else:

            unlabeled_labels = loadmat(x + "unlabeledLabels_MindReading_" + y, appendmat=True)
            unlabeled_matrix = loadmat(x + "unlabeledMatrix_MindReading" + y, appendmat=True)

        unlabeled_label = np.array(unlabeled_labels['unlabeledLabels'])
        unlabeled_label_uncertain =  unlabeled_label.T[0]
        unlabeled_label_random = unlabeled_label.T[0]
        unlabeled_mat_uncertain =  np.array(unlabeled_matrix['unlabeledMatrix'])
        unlabeled_mat_random = np.array(unlabeled_matrix['unlabeledMatrix'])


        print("unlabeled label shape: ", unlabeled_label.shape)
        print("unlabeled Mat shape: ", unlabeled_mat_random.shape)

    i=i + 1

    N = 50
    k = 10
    temp_accuracy = []
    temp_accuracy.clear()
    for i in range(N):
        # Create linear regression object
         lrmodel = linear_model.LogisticRegression()
         lrmodel.fit(trainMat_random, trainLabel_random)
         temp_accuracy.append(lrmodel.score(testMat_random, testLabel_random))
         matrixindexes = np.random.choice(np.arange(unlabeled_mat_random.shape[0]), 10, replace=False)

         for j in matrixindexes:
            # randNumber = np.random.randint(unlabeled_mat.shape[0])
            trainMat_random = np.vstack([trainMat_random, unlabeled_mat_random[j]])
            # print("unLabeled shape::", unlabeled_mat.shape)
            trainLabel_random = np.append(trainLabel_random, unlabeled_label_random[j])
            # remove from unlabeled data
         unlabeled_mat_random = np.delete(unlabeled_mat_random, matrixindexes, axis=0)
         # index = unlabeled_label[randNumber]
         unlabeled_label_random = np.delete(unlabeled_label_random, matrixindexes)

    random_accuracy_values.append(temp_accuracy)
    random_accuracy_values = np.sum(random_accuracy_values, axis=0)
    random_accuracy_values = np.true_divide(random_accuracy_values, 3)

    print("Random Accuracy: ", random_accuracy_values)



        
    

