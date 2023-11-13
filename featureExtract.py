from sklearn.preprocessing import normalize

from dataProcessing import *

features = {}
features['raw']        = X[:,0], X[:,1]
features['derivative'] = normalize(np.gradient(X[:,0], axis = 1)), normalize(np.gradient(X[:,1], axis = 1))
features['integral']   = normalize(np.cumsum(X[:,0], axis = 1)), normalize(np.cumsum(X[:,1], axis = 1))

def extract_features(*arguments):

    desired = ()
    for arg in arguments:
        desired += features[arg]

    return np.stack(desired, axis = 1)

print('Feature Extracted! ')

