from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import pandas as pd
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

# Load your cleaned dataset
data = pd.read_csv("cleaned_AfgData2.csv")

# Define your target variable and feature set
target_variable = 'targtype1'
X = data.drop(columns=[target_variable])
y = data[target_variable]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SelectKBest Feature Selection
k = 10
selector = SelectKBest(score_func=f_classif, k=k)
X_train_new = selector.fit_transform(X_train, y_train)
X_test_new = selector.transform(X_test)

# Get the selected feature names (if needed)
selected_features = X.columns[selector.get_support()]

# Initialize Decision Tree model
dt_model = DecisionTreeClassifier(criterion='gini', splitter='best')

# Train Decision Tree model
dt_model.fit(X_train_new, y_train)

# Predictions on the test set
dt_pred = dt_model.predict(X_test_new)

# Save the trained model using pickle
pickle.dump(dt_model, open('model1.pkl', 'wb'))


