import numpy as np

def get_error_rate(pred, label):
    """
    @input:
        pred: 1-D np.ndarray with dtype int. Predicted labels.
        label: 1-D np.ndarray with dtype int. The same shape as pred. Ground truth labels.
    @return:
        error rate: float. 
    """
    # TODO
    errors = 0
    for i in range(0,len(pred)):
        if(pred[i] != label[i]):
            errors=errors+1
    return float(errors) / len(pred)

