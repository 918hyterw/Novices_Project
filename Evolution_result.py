from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import os
from matplotlib import pyplot

# Evolutation part
def evolutation_function(model_path,x_train,y_train):
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    predicted_y = model_path.predict(x_train)
    predicted_y = np.array([float(round(i[0])) for i in predicted_y])

    accuracy_score_test = round(accuracy_score(y_train, predicted_y),2)
    print("Accuracy Score of the Model : ",str(int(accuracy_score_test*100))+" %")

    print("\nConfusion Matrix: \n");
    cm = confusion_matrix(y_train, predicted_y)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    pyplot.show()
    