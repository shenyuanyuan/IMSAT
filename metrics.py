import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

nmi = normalized_mutual_info_score
ari = adjusted_rand_score

def acc(y_true, y_pred, class_num = 10):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    print('print w:')
    print(w)
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    #print('print ind:')
    #print(ind)
    for j in range(class_num):
        index = np.where(ind[:, 1] == j)
        i = ind[index, 0]
        pre_true_num = w[i, j]
        real_true_num = np.sum(y_true == j)
        print('class: %d, acc rate: %7.4f' % (j, 1.0*pre_true_num/real_true_num))
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
