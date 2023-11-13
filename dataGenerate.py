from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer as LB

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


data = extract_features('raw','derivative','integral')
labels = np.array(lbl)

in_shape = data[0].shape
out_shape = tuple([1]) + in_shape

np.random.seed(10)

n_examples = labels.shape[0]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
    data, labels[:, 0], labels[:, 1].astype(int), test_size=0.2, random_state=42)

# Perform label binarization
y_train = LB().fit_transform(y_train)
y_test = LB().fit_transform(y_test)

print('Data Generated! ')