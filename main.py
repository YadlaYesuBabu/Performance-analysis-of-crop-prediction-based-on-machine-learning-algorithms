import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

# Function to introduce errors in predicted labels
def introduce_errors(y_pred, error_rate=0):
    num_errors = int(len(y_pred) * error_rate)
    indices = np.random.choice(len(y_pred), num_errors, replace=False)
    for idx in indices:
        y_pred[idx] = np.random.choice([i for i in range(len(crops)) if i != int(y_pred[idx])])
    return y_pred

st.title("Crop Prediction :rice: :corn:")
navigation = st.sidebar.radio(label="Select Action", options=["Prediction"])

crops = ['apple', 'banana', 'blackgram', 'chickpea', 'coconut', 'coffee',
         'cotton', 'grapes', 'jute', 'kidneybeans', 'lentil', 'maize',
         'mango', 'mothbeans', 'mungbean', 'muskmelon', 'orange', 'papaya',
         'pigeonpeas', 'pomegranate', 'rice', 'watermelon']

soli_data = ['Red Sandy Loam Soil', 'Clay Loam Soil', 'Saline Coastal Alluvium Soil',
             'Non Calcareous Red Soil', 'Non Calcareous Brown Soil', 'Calcareous Black Soil',
             'Red Loamy Soil', 'Black Soil', 'Red Loamy(New Delta) Soil',
             'Alluvium(Old Delta) Soil', 'Coastal Alluvium Soil',
             'Deep Red Soil', 'Saline Coastal Soil', 'Alluvium Soil',
             'Deep Red Loam Soil', 'Lateritic Soil']  # Fixed typo

if navigation == "Prediction":
    st.header("\nEnter Values to Predict the Optimal Crop")

    # Modify the file path according to your data.csv location
    file_path = "large_data.csv"
    df = pd.read_csv(file_path)
    df = df.sample(frac=1).reset_index(drop=True)

    X = df.drop("crop", axis=1)
    y = df.crop

    ordinal_enc = OrdinalEncoder()
    y = ordinal_enc.fit_transform(y.values.reshape(-1, 1))

    num_attributes = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
    cat_attributes = ["soil"]

    num_pipeline = Pipeline([
        ("std_scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder())
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes)])

    X = full_pipeline.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf_nb = GaussianNB()
    clf_nb.fit(X_train, y_train.ravel())

    clf_dt = DecisionTreeClassifier()
    clf_dt.fit(X_train, y_train.ravel())

    clf_rf = RandomForestClassifier()
    clf_rf.fit(X_train, y_train.ravel())

    clf_svm = SVC()
    clf_svm.fit(X_train, y_train.ravel())

    clf_knn = KNeighborsClassifier(n_neighbors=3)
    clf_knn.fit(X_train, y_train.ravel())

    clf_mlp = MLPClassifier(random_state=42)
    clf_mlp.fit(X_train, y_train.ravel())

    clf_adaboost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42)
    clf_adaboost.fit(X_train, y_train.ravel())

    n = st.number_input("Nitrogen Ratio", min_value=0.00)
    p = st.number_input("Phosphorous Ratio", min_value=0.00)
    k = st.number_input("Potassium Ratio", min_value=0.00)
    temperature = st.number_input("Temperature(Celsius)", min_value=0.00)
    humidity = st.number_input("Humidity", min_value=0.00)
    ph = st.number_input("pH of the Soil", min_value=1.000000, max_value=14.000000)
    rainfall = st.number_input("Rainfall", min_value=0.00)
    soil_type = st.selectbox('Soil Type', soli_data)
    var = soli_data.index(soil_type)
    inputs = np.array([[n, p, k, temperature, humidity, ph, rainfall, var]])

    # Prediction
    prediction_nb = clf_nb.predict(inputs)
    pred_dt = clf_dt.predict(inputs)
    pred_rf = clf_rf.predict(inputs)
    pred_svm = clf_svm.predict(inputs)
    pred_knn = clf_knn.predict(inputs)
    pred_mlp = clf_mlp.predict(inputs)
    pred_adaboost = clf_adaboost.predict(inputs)

    index_nb = int(prediction_nb[0])
    ind_dt = int(pred_dt[0])
    ind_rf = int(pred_rf[0])
    ind_svm = int(pred_svm[0])
    ind_knn = int(pred_knn[0])
    ind_mlp = int(pred_mlp[0])
    ind_adaboost = int(pred_adaboost[0])

    crop_nb = crops[index_nb]
    crop_dt = crops[ind_dt]
    crop_rf = crops[ind_rf]
    crop_svm = crops[ind_svm]
    crop_knn = crops[ind_knn]
    crop_mlp = crops[ind_mlp]
    crop_adaboost = crops[ind_adaboost]

    if st.button("Predict DT"):
        start_time_train = time.time()
        clf_dt.fit(X_train, y_train.ravel())
        training_time = time.time() - start_time_train

        start_time_pred = time.time()
        y_pred_dt = introduce_errors(clf_dt.predict(X_test), error_rate=0)
        prediction_time = time.time() - start_time_pred

        accuracy_dt = accuracy_score(y_test, y_pred_dt)
        precision_dt = precision_score(y_test, y_pred_dt, average='weighted')
        recall_dt = recall_score(y_test, y_pred_dt, average='weighted')
        f1_dt = f1_score(y_test, y_pred_dt, average='weighted')

        st.success("Predicted crop (DT): " + crop_dt)
        st.info(f"DT Accuracy: {accuracy_dt*100:.2f}, Precision: {precision_dt*100:.2f}, Recall: {recall_dt*100:.2f}, F1 Score: {f1_dt*100:.2f}")
        st.info(f"Training Time (DT): {training_time*1000:.4f} milli seconds")
        st.info(f"Prediction Time (DT): {prediction_time*1000:.4f} milli seconds")

        # Confusion Matrix for Decision Tree
        """st.subheader("Confusion Matrix (Decision Tree)")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        st.text("True Positive: " + str(cm_dt[0, 0]))
        st.text("False Positive: " + str(cm_dt[0, 1]))
        st.text("False Negative: " + str(cm_dt[1, 0]))
        st.text("True Negative: " + str(cm_dt[1, 1]))"""

    elif st.button("Predict NB"):
        start_time_train_nb = time.time()
        clf_nb.fit(X_train, y_train.ravel())
        training_time_nb = time.time() - start_time_train_nb

        start_time_pred_nb = time.time()
        y_pred_nb = introduce_errors(clf_nb.predict(X_test), error_rate=0)
        prediction_time_nb = time.time() - start_time_pred_nb

        accuracy_nb = accuracy_score(y_test, y_pred_nb)
        precision_nb = precision_score(y_test, y_pred_nb, average='weighted')
        recall_nb = recall_score(y_test, y_pred_nb, average='weighted')
        f1_nb = f1_score(y_test, y_pred_nb, average='weighted')

        st.success("Predicted crop (NB): " + crop_nb)
        st.info(f"NB Accuracy: {accuracy_nb*100:.2f}, Precision: {precision_nb*100:.2f}, Recall: {recall_nb*100:.2f}, F1 Score: {f1_nb*100:.2f}")
        st.info(f"Training Time (NB): {training_time_nb*1000:.4f} milli seconds")
        st.info(f"Prediction Time (NB): {prediction_time_nb*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for Naive Bayes
        cm_nb = confusion_matrix(y_test, y_pred_nb)

        """st.subheader("Confusion Matrix (Naive Bayes)")
        st.text("True Positive: " + str(cm_nb[1, 1]))
        st.text("False Positive: " + str(cm_nb[0, 1]))
        st.text("False Negative: " + str(cm_nb[1, 0]))
        st.text("True Negative: " + str(cm_nb[0, 0]))"""


    elif st.button("Predict RF"):
        start_time_train_rf = time.time()
        clf_rf.fit(X_train, y_train.ravel())
        training_time_rf = time.time() - start_time_train_rf

        start_time_pred_rf = time.time()
        y_pred_rf = introduce_errors(clf_rf.predict(X_test), error_rate=0)
        prediction_time_rf = time.time() - start_time_pred_rf

        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
        recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
        f1_rf = f1_score(y_test, y_pred_rf, average='weighted')

        st.success("Predicted crop (RF): " + crop_rf)
        st.info(f"RF Accuracy: {accuracy_rf*100:.2f}, Precision: {precision_rf*100:.2f}, Recall: {recall_rf*100:.2f}, F1 Score: {f1_rf*100:.2f}")
        st.info(f"Training Time (RF): {training_time_rf*1000:.4f} milli seconds")
        st.info(f"Prediction Time (RF): {prediction_time_rf*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for Random Forest
        cm_rf = confusion_matrix(y_test, y_pred_rf)

        """st.subheader("Confusion Matrix (Random Forest)")
        st.text("True Positive: " + str(cm_rf[1, 1]))
        st.text("False Positive: " + str(cm_rf[0, 1]))
        st.text("False Negative: " + str(cm_rf[1, 0]))
        st.text("True Negative: " + str(cm_rf[0, 0]))"""

    elif st.button("Predict SVM"):
        start_time_train_svm = time.time()
        clf_svm.fit(X_train, y_train.ravel())
        training_time_svm = time.time() - start_time_train_svm

        start_time_pred_svm = time.time()
        y_pred_svm = introduce_errors(clf_svm.predict(X_test), error_rate=0)
        prediction_time_svm = time.time() - start_time_pred_svm

        accuracy_svm = accuracy_score(y_test, y_pred_svm)
        precision_svm = precision_score(y_test, y_pred_svm, average='weighted')
        recall_svm = recall_score(y_test, y_pred_svm, average='weighted')
        f1_svm = f1_score(y_test, y_pred_svm, average='weighted')

        st.success("Predicted crop (SVM): " + crop_svm)
        st.info(f"SVM Accuracy: {accuracy_svm*100:.2f}, Precision: {precision_svm*100:.2f}, Recall: {recall_svm*100:.2f}, F1 Score: {f1_svm*100:.2f}")
        st.info(f"Training Time (SVM): {training_time_svm*1000:.4f} milli seconds")
        st.info(f"Prediction Time (SVM): {prediction_time_svm*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for Support Vector Machine
        cm_svm = confusion_matrix(y_test, y_pred_svm)

        """st.subheader("Confusion Matrix (Support Vector Machine)")
        st.text("True Positive: " + str(cm_svm[1, 1]))
        st.text("False Positive: " + str(cm_svm[0, 1]))
        st.text("False Negative: " + str(cm_svm[1, 0]))
        st.text("True Negative: " + str(cm_svm[0, 0]))"""


    elif st.button("Predict KNN"):
        start_time_train_knn = time.time()
        clf_knn.fit(X_train, y_train.ravel())
        training_time_knn = time.time() - start_time_train_knn

        start_time_pred_knn = time.time()
        y_pred_knn = introduce_errors(clf_knn.predict(X_test), error_rate=0)
        prediction_time_knn = time.time() - start_time_pred_knn

        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        precision_knn = precision_score(y_test, y_pred_knn, average='weighted')
        recall_knn = recall_score(y_test, y_pred_knn, average='weighted')
        f1_knn = f1_score(y_test, y_pred_knn, average='weighted')

        st.success("Predicted crop (KNN): " + crop_knn)
        st.info(f"KNN Accuracy: {accuracy_knn*100:.2f}, Precision: {precision_knn*100:.2f}, Recall: {recall_knn*100:.2f}, F1 Score: {f1_knn*100:.2f}")
        st.info(f"Training Time (KNN): {training_time_knn*1000:.4f} milli seconds")
        st.info(f"Prediction Time (KNN): {prediction_time_knn*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for K-Nearest Neighbors
        cm_knn = confusion_matrix(y_test, y_pred_knn)

        """st.subheader("Confusion Matrix (K-Nearest Neighbors)")
        st.text("True Positive: " + str(cm_knn[1, 1]))
        st.text("False Positive: " + str(cm_knn[0, 1]))
        st.text("False Negative: " + str(cm_knn[1, 0]))
        st.text("True Negative: " + str(cm_knn[0, 0]))"""


    elif st.button("Predict MLP"):
        start_time_train_mlp = time.time()
        clf_mlp.fit(X_train, y_train.ravel())
        training_time_mlp = time.time() - start_time_train_mlp

        start_time_pred_mlp = time.time()
        y_pred_mlp = introduce_errors(clf_mlp.predict(X_test), error_rate=0)
        prediction_time_mlp = time.time() - start_time_pred_mlp

        accuracy_mlp = accuracy_score(y_test, y_pred_mlp)
        precision_mlp = precision_score(y_test, y_pred_mlp, average='weighted')
        recall_mlp = recall_score(y_test, y_pred_mlp, average='weighted')
        f1_mlp = f1_score(y_test, y_pred_mlp, average='weighted')

        st.success("Predicted crop (MLP): " + crop_mlp)
        st.info(f"MLP Accuracy: {accuracy_mlp*100:.2f}, Precision: {precision_mlp*100:.2f}, Recall: {recall_mlp*100:.2f}, F1 Score: {f1_mlp*100:.2f}")
        st.info(f"Training Time (MLP): {training_time_mlp*1000:.4f} milli seconds")
        st.info(f"Prediction Time (MLP): {prediction_time_mlp*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for Multilayer Perceptron
        cm_mlp = confusion_matrix(y_test, y_pred_mlp)

        """st.subheader("Confusion Matrix (Multilayer Perceptron)")
        st.text("True Positive: " + str(cm_mlp[1, 1]))
        st.text("False Positive: " + str(cm_mlp[0, 1]))
        st.text("False Negative: " + str(cm_mlp[1, 0]))
        st.text("True Negative: " + str(cm_mlp[0, 0]))"""

    elif st.button("Predict AdaBoost"):
        start_time_train_adaboost = time.time()
        clf_adaboost.fit(X_train, y_train.ravel())
        training_time_adaboost = time.time() - start_time_train_adaboost

        start_time_pred_adaboost = time.time()
        y_pred_adaboost = introduce_errors(clf_adaboost.predict(X_test), error_rate=0)
        prediction_time_adaboost = time.time() - start_time_pred_adaboost

        accuracy_adaboost = accuracy_score(y_test, y_pred_adaboost)
        precision_adaboost = precision_score(y_test, y_pred_adaboost, average='weighted')
        recall_adaboost = recall_score(y_test, y_pred_adaboost, average='weighted')
        f1_adaboost = f1_score(y_test, y_pred_adaboost, average='weighted')

        st.success("Predicted crop (AdaBoost): " + crop_adaboost)
        st.info(f"AdaBoost Accuracy: {accuracy_adaboost*100:.2f}, Precision: {precision_adaboost*100:.2f}, Recall: {recall_adaboost*100:.2f}, F1 Score: {f1_adaboost*100:.2f}")
        st.info(f"Training Time (AdaBoost): {training_time_adaboost*1000:.4f} milli seconds")
        st.info(f"Prediction Time (AdaBoost): {prediction_time_adaboost*1000:.4f} milli seconds")

        # Compute and display the confusion matrix for AdaBoost
        cm_adaboost = confusion_matrix(y_test, y_pred_adaboost)

        """st.subheader("Confusion Matrix (AdaBoost)")
        st.text("True Positive: " + str(cm_adaboost[1, 1]))
        st.text("False Positive: " + str(cm_adaboost[0, 1]))
        st.text("False Negative: " + str(cm_adaboost[1, 0]))
        st.text("True Negative: " + str(cm_adaboost[0, 0]))"""