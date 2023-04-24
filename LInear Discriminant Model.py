import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pickle

# Load the EMG data set into a pandas dataframe
emg_data = pd.read_csv('/content/ML_Pattern_Recognition_Data V2.csv')

# Split the data set into features (X) and labels (y)
X = emg_data.iloc[:, 1:]
y = emg_data.iloc[:, 0]

# Create the LDA model
clf = LinearDiscriminantAnalysis()

# Train the model on the EMG data
clf.fit(X, y)

# Predict labels for the training set
y_pred = clf.predict(X)

# Calculate accuracy of the model on the training set
accuracy = accuracy_score(y, y_pred)
print("Accuracy on training set: {:.2f}%".format(accuracy * 100))

# Save the model as a pickle file
filename = 'emg_model.pkl'
pickle.dump(clf, open(filename, 'wb'))
