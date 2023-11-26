from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np


def confusion_matrix_cal(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = metrics.confusion_matrix(y_true, y_pred)

    print("Confusion Matrix:")
    print(cm)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[False, True])

    cm_display.plot()
    plt.show()

