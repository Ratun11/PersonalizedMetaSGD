import keras
from pandas import DataFrame as df
from modelEvaluation import *
from dataGenerate import *

ann_model = keras.models.load_model('ann.h5')
y_pred_ann = ann_model.predict(X_test)

result = print_results(y_pred_ann, y_test, snr_test)
print(result)
plot_results(y_pred_ann, y_test, snr_test)
plot_confusion_matrix(y_test, y_pred_ann, mods)