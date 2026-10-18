import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from keras import layers, models
from keras.callbacks import EarlyStopping
from keras.datasets import mnist

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data")
    train_path = os.path.join(data_dir, "mnist_train.csv")
    test_path = os.path.join(data_dir, "mnist_test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("[INFO] Loading MNIST dataset from local CSV files...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        x_train = train_df.iloc[:, 1:].values.astype("float32") / 255.0
        y_train = train_df.iloc[:, 0].values
        x_test = test_df.iloc[:, 1:].values.astype("float32") / 255.0
        y_test = test_df.iloc[:, 0].values
    else:
        print("[INFO] Loading MNIST dataset via Keras...")
        (x_train_raw, y_train), (x_test_raw, y_test) = mnist.load_data()
        x_train = (x_train_raw.reshape(-1, 784)).astype("float32") / 255.0
        x_test = (x_test_raw.reshape(-1, 784)).astype("float32") / 255.0

    return x_train, y_train, x_test, y_test

def benchmark_models(x_train, y_train, x_test, y_test, subset_size=15000):
    results = {}
    x_sub = x_train[:subset_size]
    y_sub = y_train[:subset_size]

    # 1. Logistic Regression
    print("\n[1/6] Running Logistic Regression...")
    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(x_sub, y_sub)
    lr_pred = lr.predict(x_test)
    results["Logistic Regression"] = accuracy_score(y_test, lr_pred) * 100
    print(f"Accuracy: {results['Logistic Regression']:.2f}%")

    # 2. Linear Regression (Baseline)
    print("\n[2/6] Running Linear Regression (Baseline)...")
    lin = LinearRegression()
    lin.fit(x_sub, y_sub)
    lin_pred = np.round(lin.predict(x_test)).clip(0, 9).astype(int)
    results["Linear Regression"] = accuracy_score(y_test, lin_pred) * 100
    print(f"Accuracy: {results['Linear Regression']:.2f}%")

    # Dimensionality Reduction with PCA (50 components)
    print("\n[INFO] Fitting PCA (50 components)...")
    pca = PCA(n_components=50, random_state=42)
    x_sub_pca = pca.fit_transform(x_sub)
    x_test_pca = pca.transform(x_test)
    print(f"Explained Variance (50 components): {pca.explained_variance_ratio_.sum() * 100:.2f}%")

    # 3. KNN with PCA
    print("\n[3/6] Running KNN (PCA-50)...")
    knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski")
    knn.fit(x_sub_pca, y_sub)
    knn_pred = knn.predict(x_test_pca)
    results["KNN (PCA-50)"] = accuracy_score(y_test, knn_pred) * 100
    print(f"Accuracy: {results['KNN (PCA-50)']:.2f}%")

    # 4. KMeans with PCA (Unsupervised + Majority Voting)
    print("\n[4/6] Running K-Means (PCA-50)...")
    kmeans = KMeans(n_clusters=10, max_iter=300, random_state=42, n_init=10)
    train_clusters = kmeans.fit_predict(x_sub_pca)
    test_clusters = kmeans.predict(x_test_pca)

    cluster_mapping = {}
    for c in range(10):
        cluster_indices = np.where(train_clusters == c)[0]
        if len(cluster_indices) > 0:
            majority_label = np.argmax(np.bincount(y_sub[cluster_indices], minlength=10))
            cluster_mapping[c] = majority_label
        else:
            cluster_mapping[c] = -1

    kmeans_pred = np.array([cluster_mapping[c] for c in test_clusters])
    results["K-Means (PCA-50)"] = accuracy_score(y_test, kmeans_pred) * 100
    print(f"Accuracy: {results['K-Means (PCA-50)']:.2f}%")

    # 5. Multilayer Perceptron (MLP)
    print("\n[5/6] Training Deep MLP Neural Network...")
    mlp = models.Sequential([
        layers.Input(shape=(784,)),
        layers.Dense(256, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])
    mlp.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    mlp.fit(x_train, y_train, batch_size=128, epochs=15, validation_split=0.2, callbacks=[early_stop], verbose=1)
    _, mlp_acc = mlp.evaluate(x_test, y_test, verbose=0)
    results["MLP (Dense ANN)"] = mlp_acc * 100
    print(f"Accuracy: {results['MLP (Dense ANN)']:.2f}%")

    # 6. Convolutional Neural Network (CNN)
    print("\n[6/6] Training Convolutional Neural Network (CNN)...")
    cnn_x_train = x_train.reshape(-1, 28, 28, 1)
    cnn_x_test = x_test.reshape(-1, 28, 28, 1)

    cnn = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax")
    ])
    cnn.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    cnn_early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    cnn.fit(cnn_x_train, y_train, batch_size=128, epochs=15, validation_split=0.2, callbacks=[cnn_early_stop], verbose=1)
    _, cnn_acc = cnn.evaluate(cnn_x_test, y_test, verbose=0)
    results["CNN (ConvNet)"] = cnn_acc * 100
    print(f"Accuracy: {results['CNN (ConvNet)']:.2f}%")

    # Summary Display
    print("\n" + "=" * 50)
    print("        MNIST BENCHMARK FINAL RESULTS")
    print("=" * 50)
    summary_df = pd.DataFrame(list(results.items()), columns=["Algorithm", "Test Accuracy (%)"])
    print(summary_df.to_string(index=False))
    print("=" * 50)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_data()
    benchmark_models(x_train, y_train, x_test, y_test)
