import numpy as np
import pandas as pd
import random
import time
import math
import matplotlib.pyplot as plt

datasets = ['A', 'B', 'usps']

#Reads and randomly splits the file. "r" represents the fraction of data we want for training.
def read_file(file, r):
    data_file = file + '.csv'
    label_file = 'labels-' +file+ '.csv'
    data = pd.read_csv(data_file, header = None).values
    labels = pd.read_csv(label_file, header = None).values
    n = len(data)
    '''
    label_1 = 0
    label_0 = 0
    for i in range(n):
        if labels[i] == 0:
            label_0 +=1
        else:
            label_1 += 1
    print(label_0, label_1)
    It was found that there is an almost even split in all datasets. Thus we don't need stratified sampling.
    '''
    i = [i for i in range(n)]
    random.shuffle(i)
    test_i = i[0:int(n/3.0)]
    train_i = i[int(n/3.0): int((1.0/3.0 + r*2.0/3.0)*n)]
    train_data = data[train_i]
    test_data = data[test_i]
    train_label = labels[train_i]
    test_label = labels[test_i]
    return train_data, test_data, train_label, test_label

#Generative model implementation as given in PRML. Numpy arrays have been used for quick calculations using inbuilt functions.
def generative(filename, size):
    train_data, test_data, train_label, test_label = read_file(filename, size)
    N = len(train_data)

    c1_ind = np.where(train_label == 1)[0]
    train_1 = train_data[c1_ind]
    mu_1 = train_1.mean(0)
    N_1 = train_1.shape[0]
    p_1 = float(N_1) / N

    c2_ind = np.where(train_label == 0)[0]
    train_2 = train_data[c2_ind]
    mu_2 = train_2.mean(0)
    p_2 = 1 - p_1

    S1 = np.cov(train_1.transpose())
    S2 = np.cov(train_2.transpose())

    S = p_1 * S1 + p_2 * S2
    S_inv = np.linalg.inv(S)

    w = np.dot(S_inv, mu_1 - mu_2)
    w0 = - 0.5 * np.dot(mu_1, np.dot(S_inv, mu_1)) + 0.5 * np.dot(mu_2, np.dot(S_inv, mu_2)) + np.log(p_1 / p_2)

    tp = tn = fp = fn = 0
    for k in range(len(test_data)):
        a = w0 + np.dot(w, test_data[k])
        sigma = 1 / (1 + np.exp(-a))
        if sigma >= 0.5:
            if test_label[k] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if test_label[k] == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp+tn)/ (tp+fp+tn+fn)
    return acc

#Special read file for irls dataset as we need no split in that.
def read_all_file(file):
    data_file = file + '.csv'
    label_file = 'labels-' +file+ '.csv'
    data = pd.read_csv(data_file, header = None).values
    labels = pd.read_csv(label_file, header = None).values
    return data, labels

#Function to compute the accuracy of the models
def compute_acc(N_t, phi_test, test_label, SN, w):
    tp = tn = fp = fn = 0
    for i in range(N_t):
        mu_a = phi_test[i].dot(w)
        sigma_a_squared = phi_test[i].T.dot(SN.dot(phi_test[i]))
        kappa = (1 + np.pi * sigma_a_squared / 8) ** (-0.5)
        p = 1.0 / (1 + np.exp(- kappa * mu_a))
        if p >= 0.5:
            if test_label[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if test_label[i] == 0:
                tn += 1
            else:
                fn += 1
    acc = (tp + tn) / (tp + fp + tn + fn)
    return acc

#Bayesian Discriminative model implementation as given in PRML. Numpy arrays have been used for quick calculations using inbuilt functions.
def bayesian(filename, size):
    if filename == 'irlstest':
        all_data, labels = read_all_file(filename)
        train_data = test_data = all_data
        train_label = test_label = labels
    else:
        train_data, test_data, train_label, test_label = read_file(filename, size)
    t = train_label
    N = len(train_data)
    ones = np.array([[1]] * N)
    phi = np.concatenate((ones, train_data), axis=1)
    M = len(phi[0])
    alpha = 0.1
    w = np.array([[0]] * M)
    update = 1
    n = 1
    I = np.eye(M)
    while n < 100 and update > 10 ** -3:
        w_old = w
        a = phi.dot(w_old)
        y = 1.0 / (1 + np.exp(-a))
        r = y * (1 - y)
        R = np.diag(r.ravel())
        temp1 = phi.T.dot(y - t) + alpha * w_old
        temp2 = alpha * I + phi.T.dot(R.dot(phi))
        w_new = w_old - np.linalg.inv(temp2).dot(temp1)
        update = np.linalg.norm(w_new - w_old) / np.linalg.norm(w_old)
        w = w_new
        n += 1
    a = phi.dot(w)
    y = 1.0 / (1 + np.exp(-a))
    SN_inv = alpha * I
    for n in range(N):
        SN_inv += y[n] * (1 - y[n]) * np.outer(phi[n], phi[n])
    SN = np.linalg.inv(SN_inv)
    N_t = len(test_data)
    ones = np.array([[1]] * N_t)
    phi_test = np.concatenate((ones, test_data), axis=1)
    acc = compute_acc(N_t, phi_test, test_label, SN, w)
    if filename == 'irlstest':
        return acc, w
    else:
        return acc

#Verifying weights for irls dataset.
acc, w = bayesian("irlstest", 1)
print("Verifying Bayesian model for irls dataset")
print("Accuracy: " ,acc)
print("Weights: ")
print(w)

#Function to run the models 30 times and get an average with standard deviation over all the runs.
def plot_learning_curve(filename):
    all_sizes = np.array([0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1])
    bay_all = []
    bay_std = []
    gen_all = []
    gen_std = []
    for size in all_sizes:
        acc_bay = []
        acc_gen = []
        for k in range(30):
            acc_bay.append(bayesian(filename, size))
            acc_gen.append(generative(filename, size))
        bay_all.append(np.mean(acc_bay))
        bay_std.append(np.std(acc_bay))
        gen_all.append(np.mean(acc_gen))
        gen_std.append(np.std(acc_gen))
    return all_sizes, gen_all, gen_std, bay_all, bay_std

#Plotting
for dataset in datasets:
    sizes, gen_all, gen_std, bay_all, bay_std = plot_learning_curve(dataset)
    print("Dataset :" +dataset)
    print("Sizes:")
    print(sizes)
    print("Accuracies and standard deviation for generative model:")
    print(gen_all)
    print(gen_std)
    print("Accuracies and standard deviation for discriminative model:")
    print(bay_all)
    print(bay_std)

    err_gen = [1-x for x in gen_all]
    err_bay = [1-x for x in bay_all]

    plt.gcf().clear()
    (_, caps, _) = plt.errorbar(sizes,err_gen,yerr=gen_std,ecolor='r', color = 'b', capsize=20,label = "Generative Model")
    for cap in caps:
        cap.set_markeredgewidth(1)
    (_, caps, _) = plt.errorbar(sizes,err_bay,yerr=bay_std,ecolor='c', color = 'g', capsize=20, label = "Discriminative Model")
    for cap in caps:
        cap.set_markeredgewidth(1)
    plt.xlabel("Training Sizes")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("Dataset: " +dataset)
    plt.legend(loc="best")
    plt.savefig(dataset + '_p1.png')

