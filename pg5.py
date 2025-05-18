import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Step 1: Generate 100 random values in the range [0,1]
np.random.seed(42)
x = np.random.rand(100, 1)

# Step 2: Label the first 50 points based on condition
y = np.where(x[:50] <= 0.5, 'Class1', 'Class2')

print("\nTraining data set up to first 50 numbers")
for i in range(50):
    print(f"x{i+1} = {x[i][0]:.3f} → classified as {y[i]}")

# Step 3: Train KNN and classify the remaining 50 points for different k values
k_values = [1, 2, 3, 4, 5, 20, 30]

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x[:50], y)
    y_pred = knn.predict(x[50:])

    print(f"\nResults for k = {k}:")
    for i, label in enumerate(y_pred, start=51):
        print(f"x{i} = {x[i-1][0]:.3f} → Classified as {label}")

# Step 4: Visualization
plt.figure(figsize=(8, 5))

# Labeled points (first 50)
colors = ['blue' if label == 'Class1' else 'red' for label in y]
plt.scatter(x[:50], np.zeros(50), c=colors, label="Labeled Data")

# Unlabeled points (last 50)
plt.scatter(x[50:], np.zeros(50), c='black', marker='x', label="Unlabeled Data")

plt.xlabel("x values")
plt.ylabel("Classification")
plt.title("KNN Classification of Random Data Points")
plt.legend()
plt.show()


# import numpy as np 
# import matplotlib.pyplot as plt 
# from sklearn.neighbors import KNeighborsClassifier 

# # Step 1: Generate 100 random values in the range [0,1] 
# np.random.seed(42) # For reproducibility 
# x = np.random.rand(100, 1) # Generate 100 random points 

# # Step 2: Label the first 50 points 
# y = np.array(['Class1' if xi <= 0.5 else 'Class2' for xi in x[:50]]) # Label based on condition 
# print("\nTraining data set upto first 50 numbers")
# for i ,label in enumerate(y[:50],start=1):
#     print(f"x{i}={x[i-1][0]:.3f}->classified as {label}")

# # Step 3: Train KNN Classifier and classify the remaining 50 points 
# k_values = [1, 2, 3, 4, 5, 20, 30] 
# for k in k_values: 
#     knn = KNeighborsClassifier(n_neighbors=k) 
#     knn.fit(x[:50], y) # Train only on the first 50 labeled points 
#     y_pred = knn.predict(x[50:]) # Predict labels for x51 to x100 
 
#     # Step 4: Display results 
#     print(f"\nResults for k = {k}:") 
#     for i, label in enumerate (y_pred, start=51): 
#         print(f"x{i} = {x[i-1][0]:.3f} → ClassiKied as {label}") 

# # Step 5: Visualization 
# plt.figure(figsize=(8, 5)) 

# plt.scatter(x[:50], np.zeros(50), c=['blue' if yi == 'Class1' else 'red' for yi in y], 
# label="Labeled Data") 

# plt.scatter(x[50:], np.zeros(50), c='black', marker='x', label="Unlabeled Data") 

# plt.xlabel("x values") 
# plt.ylabel("Classification") 
# plt.title("KNN Classification of Random Data Points") 
# plt.legend( ) 
# plt.show( )




