import numpy as np
import  tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import jaccard_similarity_score
def sensitive(prediction,actual):
    fn = 0.
    tp = 0.
    fp = 0.
    for (i, v) in enumerate(actual):
        for (j, e) in enumerate(v):

            if (actual[i][j] == 1 and prediction[i][j] == 1):
                tp += 1
            elif (actual[i][j] == 1 and prediction[i][j] == 0):
                fn += 1
            elif (actual[i][j] == 0 and prediction[i][j] == 1):
                fp += 1
    return tp / (tp + fn)


def sen(prediction,actual):
    prediction=np.argmax(prediction,axis=3)
    actual=np.argmax(actual,axis=3)
    n_batches=actual.shape[0]
    result=0
    for i in range(n_batches):
        result+=sensitive(prediction[i],actual[i])
    result=result/n_batches
    return result
def tnr(prediction,actual):
    fn = 0.
    tp = 0.
    fp = 0.
    tn=0.
    for (i, v) in enumerate(actual):
        for (j, e) in enumerate(v):

            if (actual[i][j] == 1 and prediction[i][j] == 1):
                tp += 1
            elif (actual[i][j] == 1 and prediction[i][j] == 0):
                fn += 1
            elif (actual[i][j] == 0 and prediction[i][j] == 1):
                fp += 1
            elif (actual[i][j] == 0 and prediction[i][j] == 0):
                tn+=1
    return tn / (tn + fp)


def TNR(prediction,actual):
    prediction=np.argmax(prediction,axis=3)
    actual=np.argmax(actual,axis=3)
    n_batches=actual.shape[0]
    result=0.
    for i in range(n_batches):
        result+=tnr(prediction[i],actual[i])
    result=result/n_batches
    return result


def acc(prediction,actual):
    return np.sum(np.argmax(prediction, 3) == np.argmax(actual, 3)) /(prediction.shape[0] * prediction.shape[1] * prediction.shape[2])


def precision(prediction,actual):
    prediction=np.argmax(prediction,axis=3)
    actual=np.argmax(actual,axis=3)
    n_batches = actual.shape[0]
    result = 0.
    for i in range(n_batches):

        result += pre(prediction[i], actual[i])
    result = result / n_batches
    return result
def pre(prediction,actual):
    fn = 0.
    tp = 0.
    fp = 0.
    for (i, v) in enumerate(actual):
        for (j, e) in enumerate(v):

            if (actual[i][j] == 1 and prediction[i][j] == 1):
                tp += 1
            elif (actual[i][j] == 1 and prediction[i][j] == 0):
                fn += 1
            elif (actual[i][j] == 0 and prediction[i][j] == 1):
                fp += 1

    return tp / (tp + fp)
def f1(prediction,actual):
    fn = 0.
    tp = 0.
    fp = 0.
    for (i, v) in enumerate(actual):
        for (j, e) in enumerate(v):

            if (actual[i][j] == 1 and prediction[i][j] == 1):
                tp += 1
            elif (actual[i][j] == 1 and prediction[i][j] == 0):
                fn += 1
            elif (actual[i][j] == 0 and prediction[i][j] == 1):
                fp += 1
    return 2*tp/(2*tp+fn+fp)

def f1score2(prediction,actual):
    prediction=np.argmax(prediction,axis=3)
    actual=np.argmax(actual,axis=3)
    n_batches = actual.shape[0]
    result = 0.
    for i in range(n_batches):
        result += f1(prediction[i], actual[i])
    result = result / n_batches
    return result


def roc_Auc(prediction,actual):

    n_batches = actual.shape[0]
    result=0.
    for i in range(n_batches):
        result += roc_auc_score(actual[i][:,:,0].flatten(),prediction[i][:,:,0].flatten())
    result = result / n_batches
    return result
# def JS(prediction,actual):
#     n_batches = actual.shape[0]
#     result=0.
#     for i in range(n_batches):
#         result += jaccard_similarity_score(prediction[i].flatten(), actual[i].flatten())
#     result = result / n_batches
#     return result
