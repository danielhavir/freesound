import numpy as np
import sklearn.metrics as metrics
import json
import warnings
warnings.filterwarnings('ignore')

def print_cm(cm, labels, print_fc=print):
    # Pretty prints confusion matrix
    columnwidth = max([len(x) for x in labels] + [5])
    empty_cell = " " * columnwidth
    msg = "    " + 'Pred '
    for label in labels:
        msg += "%{0}s ".format(columnwidth) % label
    print_fc(msg)
    print_fc("    " + 'True')
    for i, label1 in enumerate(labels):
        msg = "    %{0}s ".format(columnwidth) % label1
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            msg += cell + " "
        print_fc(msg)


def print_results(true, pred, loss, phase, epoch=-1, cm=True, print_fc=print):
    mcc = metrics.matthews_corrcoef(true, pred)
    acc = metrics.accuracy_score(true, pred)
    precision = metrics.precision_score(true, pred, average='macro')
    recall = metrics.recall_score(true, pred, average='macro')
    print_fc(4*'#' + ' ' + phase.upper() + f' Epoch {epoch} ' + 4*'#')
    print_fc(f'Loss: {loss}')
    print_fc(f'Accuracy: {acc}')
    print_fc(f"Precision: {precision}")
    print_fc(f"Recall: {recall}")
    print_fc(f'MCC: {mcc}')
    if cm:
        print_fc('Confusion matrix:')
        C = metrics.confusion_matrix(true, pred)
        labels = np.sort(np.unique(true)).astype(str)
        print_cm(C, labels, print_fc=print_fc)
    print_fc(" ")
    return acc, precision, recall


def top_three(true, pred, print_fc=print):
    one = (pred[:,0]==true).astype(int)
    two = (pred[:,1]==true).astype(int)
    thr = (pred[:,2]==true).astype(int)
    print_fc(f'Top-1 Accuracy: {round(one.mean(), 3)}')
    print_fc(f'Top-2 Accuracy: {round((one+two).mean(), 3)}')
    print_fc(f'Top-3 Accuracy: {round((one+two+thr).mean(), 3)}')
