import os
import numpy as np
from scipy.io import loadmat
from sklearn import linear_model
import matplotlib.pyplot as plt

data_MMI = os.getcwd() + "/data" + "/MMI/"
data_MR = os.getcwd() + "/data" + "/MindReading/"
print("path is")
print(data_MR)
list_of_datasets = [data_MMI, data_MR]
random_accuracy = []
uncertain_accuracy = []
list_of_numbers = ["1", "2", "3"]
iterator = 0
i=0
f = open(os.getcwd() + '\ random_accuracy.txt', 'w')
f1 = open(os.getcwd() + '\ uncertain_accuracy.txt', 'w')

for x in list_of_datasets:
    data = x
    print("Here is the data")
    print(data)

    random_accuracy= []
    uncertain_accuracy = []
    for y in list_of_numbers:
        print("i is")
        print(i)
        if iterator == 1:
            trainingLabel_dataset = loadmat(x + "trainingLabels_MindReading_" + y+".mat")
            traningMat_dataset = loadmat(x + "trainingMatrix_MindReading" + y+".mat")
        else:
            trainingLabel_dataset = loadmat(x + "trainingLabels_" + y+".mat")
            traningMat_dataset = loadmat(x + "trainingMatrix_" + y+".mat")

        trainLabel = np.array(trainingLabel_dataset['trainingLabels'])
        print("trainlabel")
        print(trainLabel)
        trainLabel_uncertain_transpose = trainLabel.T
        trainLabel_uncertain=trainLabel_uncertain_transpose[0]
        trainLabel_transpose = trainLabel.T
        trainLabel=trainLabel_transpose[0]
        trainMat_uncertain = np.array(traningMat_dataset['trainingMatrix'])
        trainMat = np.array(traningMat_dataset['trainingMatrix'])
        if iterator == 1:
            testingLabel_dataset = loadmat(x + "testingLabels_MindReading" + y+".mat")
            testingMat_dataset = loadmat(x + "testingMatrix_MindReading" + y+".mat")
        else:
            testingLabel_dataset = loadmat(x + "testingLabels_" + y+".mat")
            testingMat_dataset = loadmat(x + "testingMatrix_" + y+".mat")

        testLabel= np.array(testingLabel_dataset['testingLabels'])
        testLabel_uncertain_transpose = testLabel.T
        testLabel_uncertain=testLabel_uncertain_transpose[0]
        testLabel_random_transpose = testLabel.T
        testLabel_random=testLabel_random_transpose[0]
        testMat_uncertain = np.array(testingMat_dataset['testingMatrix'])
        testMat = np.array(testingMat_dataset['testingMatrix'])


        if iterator == 1:
            unlabeled_labels = loadmat(x + "unlabeledLabels_MindReading_" + y+".mat")
            unlabeled_matrices = loadmat(x + "unlabeledMatrix_MindReading" + y+".mat")
        else:
            unlabeled_labels = loadmat(x + "unlabeledLabels_" + y +".mat")
            unlabeled_matrices = loadmat(x + "unlabeledMatrix_" + y +".mat")

        unlabeled_label = np.array(unlabeled_labels['unlabeledLabels'])
        unlabeled_label_uncertain_transpose=unlabeled_label.T
        unlabeled_label_uncertain=unlabeled_label_uncertain_transpose[0]
        unlabeled_label = unlabeled_label.T
        unlabeled_label=unlabeled_label[0]
        unlabeled_mat_uncertain = np.array(unlabeled_matrices['unlabeledMatrix'])
        unlabeled_mat = np.array(unlabeled_matrices['unlabeledMatrix'])

        import random
        N = 50
        k = 10
        step_accuracy_r = []
        a=unlabeled_mat.shape[0]
        output_shape=10
        for s in range(N):
            model_obj = linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=200)
            model_obj.fit(trainMat, trainLabel)
            score=model_obj.score(testMat, testLabel_random)
            step_accuracy_r.append(score)
            indices = np.random.choice(unlabeled_mat.shape[0], output_shape, replace=False)
            for t in indices:
                trainMat = np.vstack([trainMat, unlabeled_mat[t]])
                trainLabel = np.append(trainLabel, unlabeled_label[t])
            unlabeled_mat = np.delete(unlabeled_mat, indices, 0)
            unlabeled_label = np.delete(unlabeled_label, indices,None)

        random_accuracy.append(step_accuracy_r)
        step_accuracy_u = []
        for m in range(N):
            model_uncertain = linear_model.LogisticRegression(random_state=0,solver='lbfgs',max_iter=200)
            model_uncertain.fit(trainMat_uncertain, trainLabel_uncertain)
            score_uncertain=model_uncertain.score(testMat_uncertain, testLabel_uncertain)
            step_accuracy_u.append(score_uncertain)
            normal_probability = model_uncertain.predict_proba(unlabeled_mat_uncertain)
            log_probability = model_uncertain.predict_log_proba(unlabeled_mat_uncertain)
            entropy = []
            sorted_indices = []
            index_to_be_deleted = []
            for index_2 in range(normal_probability.shape[0]):
                val = 0
                for index_3 in range(normal_probability.shape[1]):
                    val = val + (-(normal_probability[index_2][index_3] * log_probability[index_2][index_3]))

                entropy.append(val)
            print(entropy)
            sorted_indices=np.argsort(entropy)

            size = len(sorted_indices)
            for t in range(k):
                index_1=size - t - 1
                trainMat_uncertain = np.vstack([trainMat_uncertain, unlabeled_mat_uncertain[sorted_indices[index_1]]])
                trainLabel_uncertain = np.append(trainLabel_uncertain, unlabeled_label_uncertain[sorted_indices[index_1]])
                index_to_be_deleted.append(sorted_indices[index_1])

            unlabeled_mat_uncertain = np.delete(unlabeled_mat_uncertain, index_to_be_deleted, 0)
            unlabeled_label_uncertain = np.delete(unlabeled_label_uncertain, index_to_be_deleted,None)

        uncertain_accuracy.append(step_accuracy_u)

    random_accuracy = np.sum(random_accuracy, 0)
    random_accuracy = np.divide(random_accuracy, len(list_of_numbers))


    uncertain_accuracy = np.sum(uncertain_accuracy,0)
    uncertain_accuracy = np.divide(uncertain_accuracy, len(list_of_numbers))
    if (iterator == 0):
        f.write("####### MMI ####### \n ")
    else:
        f.write("####### MindReading ####### \n ")
    for items in random_accuracy:
        f.write("%s\n" % items)

    if(iterator==0):
        f1.write("####### MMI ######\n")
    else:
        f1.write("###### MindReading ###### \n")
    for items in uncertain_accuracy:
        f1.write("%s\n" % items)

    if iterator == 1:
        print("###MindReading###")
    else:
        print("###MMI###")

    print("Random Accuracy: ", random_accuracy)
    print("Uncertain Accuracy: ", uncertain_accuracy)

    if iterator == 1:
        plt.plot(range(0, 50), random_accuracy, 'gs')
        plt.plot(range(0, 50), uncertain_accuracy, 'ro')
        plt.axis([0, 50, 0, 1])
        plt.xlabel('No. of Iterations')
        plt.ylabel('Accuracy values')
        plt.legend(('Random Sampling', 'Uncertainty Sampling'),loc='lower right')
        plt.title('Active Learning on MindReading dataset')
        plt.savefig("./Active_Learning_on_MindReading_dataset.png")
        plt.show()
    else:
        plt.plot(range(0, 50), random_accuracy, 'gs')
        plt.plot(range(0, 50), uncertain_accuracy, 'ro')
        plt.axis([0, 50, 0, 1])
        plt.xlabel('No. of Iterations')
        plt.ylabel('Accuracy values')
        plt.legend(('Random Sampling', 'Uncertainty Sampling'),loc='lower right')
        plt.title('Active Learning on MMI dataset')
        plt.savefig("./Active_Learning_on_MMI_dataset.png")
        plt.show()
    iterator = iterator + 1
f.close()
f1.close()