import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
from sklearn.model_selection import train_test_split 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn import tree 

data = load_breast_cancer() #Loads the breast cancer dataset  
X = data.data  #Contains the input features (like tumor size, texture, etc.), shape (n_samples,  n_features). 

y = data.target #Contains the target labels – 0 for malignant and 1 for benign tumors. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #Splits the data into training and test datasets. 

clf = DecisionTreeClassifier(random_state=42) #Creates an instance of a decision tree classifier. 
clf.fit(X_train, y_train)   # Trains the model on the training data. 
y_pred = clf.predict(X_test) #Uses the trained model to predict the labels for the test dataset. 
accuracy = accuracy_score(y_test, y_pred) #compares predicted values with actual labels 

print(f"Model Accuracy: {accuracy * 100:.2f}%") 

#Predicting a new Sample 
new_sample = np.array([X_test[0]])  
prediction = clf.predict(new_sample) 
prediction_class = "Benign" if prediction == 1 else "Malignant"     
                      
print(f"Predicted Class for the new sample: {prediction_class}") 

# Visualizing the Decision Tree 
plt.figure(figsize=(12,8)) 
tree.plot_tree(clf, filled=True, feature_names=data.feature_names, class_names=data.target_names) 
plt.title("Decision Tree - Breast Cancer Dataset") 
plt.show()