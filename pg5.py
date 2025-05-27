import numpy as np 
import matplotlib.pyplot as plt

np.random.seed(42) 
X=np.linspace(-3,3,100) 
Y=np.sin(X)+np.random.normal(0,0.1,100) 

X_matrix=np.c_[np.ones(X.shape[0]),X]

def get_weights(query_point,X,tau):
    return np.exp(-((X-query_point)**2)/ (2*tau**2))

def locally_weighted_regression(query_x,X,Y,tau):
    W=np.diag(get_weights(query_x,X,tau))
    theta=np.linalg.pinv(X_matrix.T@W@X_matrix)@X_matrix.T@W@Y
    return np.array([1,query_x])@theta

tau_values=[0.1,0.3,1.0] 
plt.figure(figsize=(10,6))

for tau in tau_values:
    Y_pred = np.array([locally_weighted_regression(x,X,Y,tau)for x in X ])
    plt.plot(X,Y_pred,label=f'Tau={tau}')
    plt.scatter(X,Y,color='blue',label='Original Data',alpha=0.5)
plt.xlabel("X")
plt.ylabel("Y") 
plt.title("Locally Weighted Regression with different Tou value")
plt.legend()
plt.show()   
