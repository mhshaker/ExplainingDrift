import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import Uncertainty as unc
import UncertaintyM as uncM
import matplotlib.pyplot as plt

import data_provider as dp

# load the data
features_all, targets_all = dp.load_data("./Data/")
features_list, targets_list = dp.partition_data(features_all, targets_all,100)

features = features_list[0]
targets = targets_list[0]

print("First block features shape ", features.shape)
print("First block targets shape ", targets.shape)

# split and shuffel the data
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, shuffle=False, random_state=1)

# train the model
model = RandomForestClassifier(max_depth=10, n_estimators=10, random_state=1)
model.fit(X_train, y_train[:,0]) # remove keys when fiting the model

predictions = model.predict(X_test)
print("model test score = ", model.score(X_test, y_test[:,0]))

# Aleatoric uncertianty for X_test
total_uncertainty, epistemic_uncertainty, aleatoric_uncertainty = unc.model_uncertainty(model, X_test, X_train, y_train[:,0])

# AR plot
avg_acc, avg_min, avg_max, avg_random ,steps = uncM.accuracy_rejection2(predictions.reshape((1,-1)), y_test[:,0].reshape((1,-1)), total_uncertainty.reshape((1,-1)))
plt.plot(steps, avg_acc*100)
plt.savefig(f"./AR_plot.png",bbox_inches='tight')