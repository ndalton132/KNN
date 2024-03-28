import sklearn
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml


# def euclidean_distance(x1, x2):
#     x1 = np.array(x1, dtype=float)
#     x2 = np.array(x2, dtype=float)
#     return np.sqrt(np.sum(np.square(x1 - x2)))

# def main():
#     mnist = load_mnist_dataset("C:\\Users\\nickd\\OneDrive\\Desktop\\School\\AI\\KNN\\MNIST")
    
#     X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)
#     for i in range(2,20,2):
#         KNN(X_train, X_test, y_train, y_test,i)

    
def trainData(X_train, y_train, k):
    count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
    dataList = []

    while True:
        finished = True

        if all(count[key] == k for key in count):
            return dataList

        for i in range(len(X_train)):
            pixelType = int(y_train.iloc[i])
            if count[pixelType] < k:
                dataList.append((X_train.iloc[i].values, pixelType))
           
                count[pixelType] += 1



def KNN(X_train, X_test, y_train, y_test,k):
    
    list = trainData(X_train,y_train,k)
    
    correct = 0
    incorrect = 0
    
    print(len(y_test))
    
    for i in range(0,500):
        countInner = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        for j in range(0,len(list)):
            
            #count distances and take the lowest value from count inner
        
            pixelValues = X_test.iloc[i].values
            pixelType = y_test.iloc[i]
            
            listVal = list[j][0]
            
            distance_sum = 0
            for p in range(len(pixelValues)):
                distance_sum += (listVal[p] - pixelValues[p]) ** 2  # Euclidean distance calculation

            # Add the sum of distances to countInner
            countInner[list[j][1]] = np.sqrt(distance_sum)
            
        min_key = min(countInner, key=countInner.get)
        if int(pixelType) == min_key:
            correct += 1
            #print("Correct!", pixelType, "Actual:", min_key)
        else:
            incorrect += 1
            #print("Inncorect!", pixelType, "Actual:", min_key)
    print("Accuracy for k =",k,": ", correct / (correct + incorrect))
            
            
        
        
            

    
    

def load_mnist_dataset(file_path):
    

    if not os.path.exists(file_path):
        raise FileNotFoundError("The specified file does not exist.")
    
    with open(file_path, 'rb') as f:
        mnist = pickle.load(f)
    
    print("MNIST dataset loaded from:", file_path)
    return mnist

#Used this tutorial
#https://www.datacamp.com/tutorial/k-nearest-neighbor-classification-scikit-learn
#mnist = load_mnist_dataset("C:\\Users\\nickd\\OneDrive\\Desktop\\School\\AI\\KNN\\MNIST")


mnist = fetch_openml('mnist_784', version=1)


for i in range(1,10):

    X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.2, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy for k=",i,":", accuracy)


 