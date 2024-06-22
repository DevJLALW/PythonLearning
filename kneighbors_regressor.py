from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor

data = datasets.load_diabetes()
#
print('Data ',data)
print(data.feature_names)
print("Shape of data:")
print(data.data.shape)

# Print the first few rows of the data and target
print("First few rows of data:")
for i in range(min(5, data.data.shape[0])):
    print(data.data[i])

data_train, data_test, target_train, target_test = train_test_split(data.data,data.target, train_size=0.8, random_state=50)


print("First few rows of target:")
print(data.target[:5])
print(data_train[0])
scaler = preprocessing.MinMaxScaler()
scaler = scaler.fit(data_train)
data_scaled_train = scaler.transform(data_train)
data_scaled_test = scaler.transform(data_test)
print(data_scaled_train[0])

pca = PCA(n_components=4)
pca.fit(data_scaled_train)
data_lower_dim_train = pca.transform(data_scaled_train)
data_lower_dim_test = pca.transform(data_scaled_test)
print("PCA ",pca.explained_variance_)

model = KNeighborsRegressor(n_neighbors=25)
model.fit(data_lower_dim_train, target_train)
result = model.predict(data_lower_dim_test)
score=mean_squared_error(target_test,result)
print(result)
print(target_test[:20])


best_accuracy = float('inf')
best_n_neighbors=0
print("Varying n_neighbors")
for i in range(1, 50):  # Start from 1 to avoid n_neighbors=0
    #model = KNeighborsClassifier(n_neighbors=i)
    model = KNeighborsRegressor(n_neighbors=i)
    model.fit(data_lower_dim_train, target_train)
    result = model.predict(data_lower_dim_test)
    score = mean_squared_error(target_test,result)
    print(f"n_neighbors={i}, mean_squared_error: {score:.2f}")

    # Update the best accuracy and the corresponding number of neighbors
    if score < best_accuracy:
        best_accuracy = score
        best_n_neighbors = i

# Print the best result
print(f"\nBest n_neighbors={best_n_neighbors} with mean_squared_error: {best_accuracy:.5f}")

#print(confusion_matrix(y_true=target_test,y_pred=result))
print('Real Value:', target_train[:3])
print('Predicted:', result[:3])

#print(score)
# 1 gave .86, 50 - 0.6, 20 - 0.63, 15 -0.69, 25 - 0.77
