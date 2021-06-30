#!/usr/bin/env python
# coding: utf-8
import os

datapath = os.getcwd()

MMI_datapath = datapath + "/data" + "/MMI/"  # MMI data folder
# MMI_datapath

MindReading_datapath = datapath + "/data" + "/MindReading/"  # only mindreading data path
# MindReading_datapath

datapath = MindReading_datapath
print(datapath)

# In[573]:


from sklearn import datasets, linear_model
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.stats import entropy

for datasetLoop in range(1, 3):
    if datasetLoop == 1:
        datapath = MindReading_datapath

    else:
        datapath = MMI_datapath
    random_accuracy_values = []
    uncertain_accuracy_values = []
    random_accuracy_values.clear()
    uncertain_accuracy_values.clear()
    for dataID in range(1, 4):
        print(dataID)
        # traing
        if datasetLoop == 1:
            trainingLabel_1 = loadmat(datapath + "trainingLabels_MindReading_" + str(dataID), appendmat=True)
            traningMat_1 = loadmat(datapath + "trainingMatrix_MindReading" + str(dataID), appendmat=True)
        else:
            trainingLabel_1 = loadmat(datapath + "trainingLabels_" + str(dataID), appendmat=True)
            traningMat_1 = loadmat(datapath + "trainingMatrix_" + str(dataID), appendmat=True)
        trainLabel = np.array(trainingLabel_1['trainingLabels'])
        trainLabel_U = trainLabel = trainLabel.T[0]
        trainMat_U = trainMat = np.array(traningMat_1['trainingMatrix'])

        print("Train Mat shape:", trainMat.shape)
        print("Train Label shape:", trainLabel.shape)

        # testing
        if datasetLoop == 1:
            testingLabel_1 = loadmat(datapath + "testingLabels_MindReading" + str(dataID), appendmat=True)
            testingMat_1 = loadmat(datapath + "testingMatrix_MindReading" + str(dataID), appendmat=True)
        else:
            testingLabel_1 = loadmat(datapath + "testingLabels_" + str(dataID), appendmat=True)
            testingMat_1 = loadmat(datapath + "testingMatrix_" + str(dataID), appendmat=True)

        testLabel = np.array(testingLabel_1['testingLabels'])
        testLabel_U = testLabel = testLabel.T[0]
        testMat_U = testMat = np.array(testingMat_1['testingMatrix'])
        # print(testMat)
        print("Test Mat shape:", testMat.shape)
        print("Test Label shape:", testLabel.shape)

        # unlabeled
        if datasetLoop == 1:
            unlabeled_L = loadmat(datapath + "unlabeledLabels_MindReading_" + str(dataID), appendmat=True)
            unlabeled_M = loadmat(datapath + "unlabeledMatrix_MindReading" + str(dataID), appendmat=True)
        else:
            unlabeled_L = loadmat(datapath + "unlabeledLabels_" + str(dataID), appendmat=True)
            unlabeled_M = loadmat(datapath + "unlabeledMatrix_" + str(dataID), appendmat=True)

        unlabeled_label = np.array(unlabeled_L['unlabeledLabels'])
        unlabeled_label_U = unlabeled_label = unlabeled_label.T[0]
        unlabeled_mat_U = unlabeled_mat = np.array(unlabeled_M['unlabeledMatrix'])

        print("unlabeled label shape: ", unlabeled_label.shape)
        print("unlabeled Mat shape: ", unlabeled_mat.shape)

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

        # ----------------------------------------------------------------/

        temp_accuracy_uncertain = []
        for i in range(N):
            lrmodel_U = linear_model.LogisticRegression()
            lrmodel_U.fit(trainMat_U, trainLabel_U)
            temp_accuracy_uncertain.append(lrmodel_U.score(testMat_U, testLabel_U))

            predict_prob = lrmodel_U.predict_proba(unlabeled_mat_U)
            # print ("predict::", predict_prob.shape)
            predict_log_prob = lrmodel_U.predict_log_proba(unlabeled_mat_U)
            # print("logprob: ", predict_log_prob.shape)
            predict_mul = []
            sorted_index = []
            predict_mul.clear()
            sorted_index.clear()
            for index_a in range(predict_prob.shape[0]):
                temp = 0
                for index_b in range(predict_prob.shape[1]):
                    temp = temp + (-1 * predict_prob[index_a][index_b] * predict_log_prob[index_a][index_b])
                predict_mul.append(temp)
            sorted_index = sorted(range(len(predict_mul)), key=lambda k: predict_mul[k])

            # print ("sorted size: ", len(sorted_index))

            del_index = []
            del_index.clear()
            size = len(sorted_index)
            for j in range(k):
                # print ("TL : ", unlabeled_mat_U.shape)
                trainMat_U = np.vstack([trainMat_U, unlabeled_mat_U[sorted_index[size - j - 1]]])
                trainLabel_U = np.append(trainLabel_U, unlabeled_label_U[sorted_index[size - j - 1]])
                # print ("TL : ", unlabeled_label_U[sorted_index[j]])
                del_index.append(sorted_index[size - j - 1])
            # remove from unlabeled data
            unlabeled_mat_U = np.delete(unlabeled_mat_U, del_index, axis=0)
            # index = unlabeled_label_U[sorted_index[j]]
            unlabeled_label_U = np.delete(unlabeled_label_U, del_index)

        uncertain_accuracy_values.append(temp_accuracy_uncertain)
        # ----------------------------------------------------------------/

    random_accuracy_values = np.sum(random_accuracy_values, axis=0)
    uncertain_accuracy_values = np.sum(uncertain_accuracy_values, axis=0)

    random_accuracy_values = np.true_divide(random_accuracy_values, 3)
    uncertain_accuracy_values = np.true_divide(uncertain_accuracy_values, 3)

    if datasetLoop == 1:
        print(":::MindReading:::")
    else:
        print(":::MMI:::")
    print("Random Accuracy: ", random_accuracy_values)
    print("Uncertain Accuracy: ", uncertain_accuracy_values)

    if datasetLoop == 1:
        plt.plot(range(0, 50), random_accuracy_values, 'ro')
        plt.plot(range(0, 50), uncertain_accuracy_values, 'bs')
        plt.axis([0, 50, 0, 1])
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.legend(('Random Sampling', 'Uncertainty Sampling'),
                   loc='upper right')
        plt.title('Active Learning on MindReading dataset')
        plt.show()
    else:
        plt.plot(range(0, 50), random_accuracy_values, 'ro')
        plt.plot(range(0, 50), uncertain_accuracy_values, 'bs')
        plt.axis([0, 50, 0, 1])
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.legend(('Random Sampling', 'Uncertainty Sampling'),
                   loc='upper right')
        plt.title('Active Learning on MMI dataset')
        plt.show()