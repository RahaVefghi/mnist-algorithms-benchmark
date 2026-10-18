import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score

from keras.datasets import mnist

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TRAIN_PATH = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_PATH = os.path.join(DATA_DIR, "mnist_test.csv")

if os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH):
  train_data = pd.read_csv(TRAIN_PATH)
  test_data = pd.read_csv(TEST_PATH)
  x_train = train_data.iloc[:, 1:].values
  y_train = train_data.iloc[:, 0].values
  x_test = test_data.iloc[:, 1:].values
  y_test = test_data.iloc[:, 0].values
else:
  print("[INFO] Loading MNIST dataset via Keras...")
  (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()
  x_train = x_train_raw.reshape(-1, 784)
  x_test = x_test_raw.reshape(-1, 784)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

SUBSET_SIZE = 15000
x_train_sub = x_train[:SUBSET_SIZE]
y_train_sub = y_train[:SUBSET_SIZE]

LR_model = LogisticRegression(max_iter=1000)
LR_model.fit(x_train, y_train)
LR_y_pred = LR_model.predict(x_test)
LR_accuracy = accuracy_score(y_test, LR_y_pred)

print(f"accuracy of logistic regression is :{LR_accuracy *100:.2f}%")


#####################################################################################
pca = PCA(n_components=50, random_state=42)
x_train_pca = pca.fit_transform(x_train_sub)
x_test_pca = pca.transform(x_test)

print(f"PCA explained variance(50 components):{pca.explained_variance_ratio_.sum()*100:.2f}%")


knn_model = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
knn_model.fit(x_train_pca, y_train_sub)
knn_y_pred = knn_model.predict(x_test_pca)
knn_accuracy = accuracy_score(y_test, knn_y_pred)

print(f"accuracy of KNN is :{knn_accuracy *100:.2f}%")


#####################################################################################################

kmeans_model = KMeans(n_clusters=10, max_iter=300, random_state=42,n_init=10)

kmeans_model.fit(x_train_pca)

train_cluster = kmeans_model.predict(x_train_pca)
test_cluster = kmeans_model.predict(x_test_pca)

mapping ={}

for cluster in range(10):
  indices_in_cluster =np.where(train_cluster == cluster)[0]
  if len(indices_in_cluster) > 0:
    
    labels = y_train_sub[indices_in_cluster]
    counts = np.bincount(labels, minlength=10)
    maximum_label = np.argmax(counts)
    
    mapping[cluster] = maximum_label
  else:
    mapping[cluster] = -1

kmeans_y_pred = np.array([mapping[cluster] for cluster in test_cluster])

kmeans_accuracy = accuracy_score(y_test, kmeans_y_pred)  
    
print(f"accuracy of KMeans is :{kmeans_accuracy *100:.2f}%")

#################################################################################################

linear_model = LinearRegression()

linear_model.fit(x_train, y_train)
linear_y_pred = linear_model.predict(x_test)

linear_y_pred = np.round(linear_y_pred).clip(0,9).astype(int)

linear_accuracy = accuracy_score(y_test, linear_y_pred)
print(f"accuracy of Linear Rgression is :{linear_accuracy *100:.2f}%")

#####################################################################

mlp_model = models.Sequential([
                              layers.Input(shape=(784,)),
                              layers.Dense(256, activation="relu"),
                              layers.BatchNormalization(),
                              layers.Dropout(0.3),
                              layers.Dense(128, activation="relu"),
                              layers.BatchNormalization(),
                              layers.Dropout(0.3),
                              layers.Dense(10, activation="softmax")
                              
])

mlp_model.summary()

mlp_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

mlp_early_stop = EarlyStopping(
  monitor="val_loss",
  patience=3,
  restore_best_weights=True
)

H = mlp_model.fit(x_train, y_train, batch_size=128, epochs=30, validation_split=0.2,callbacks=[mlp_early_stop], verbose=1)


loss, acc = mlp_model.evaluate(x_test, y_test)

print(f"loss:{loss:.2f}, accuracy:{acc*100:.2f}%")

##################################################################################33

cnn_x_train = x_train.reshape(-1, 28,28,1)
cnn_x_test = x_test.reshape(-1, 28,28,1)

model_cnn = models.Sequential([   
                              layers.Input(shape=(28,28,1)),
                              layers.Conv2D(filters=32, kernel_size=(3,3), activation="relu"),
                              layers.BatchNormalization(),
                              layers.MaxPooling2D(pool_size=(2,2)),
                              
                              layers.Conv2D(filters=64, kernel_size=(3,3), activation="relu"),
                              layers.BatchNormalization(),
                              layers.MaxPooling2D(pool_size=(2,2)),
                              layers.Flatten(),
                              layers.Dense(128, activation="relu"),
                              layers.Dropout(0.5),
                              layers.Dense(10, activation="softmax")
                              
                          
])
model_cnn.summary()
model_cnn.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cnn_early_stop = EarlyStopping(
  monitor="val_loss",
  patience=3,
  restore_best_weights=True
)

H_cnn = model_cnn.fit(
    cnn_x_train, y_train,          
    batch_size=128,
    epochs=30, 
    validation_split=0.2,
    callbacks=[cnn_early_stop],
    verbose=1
)

loss_cnn, acc_cnn = model_cnn.evaluate(cnn_x_test, y_test)
print(f"accuracy of CNN: {acc_cnn * 100:.2f}%")
print(f"Loss CNN: {loss_cnn:.4f}")

print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Logistic Regression : {LR_accuracy * 100:.2f}%")
print(f"Linear Regression : {linear_accuracy * 100:.2f}% ")
print(f"KNN (PCA-50) : {knn_accuracy * 100:.2f}%")
print(f"KMeans (PCA-50): {kmeans_accuracy * 100:.2f}%")
print(f"MLP: {acc * 100:.2f}%")
print(f"CNN: {acc_cnn * 100:.2f}%")
 
