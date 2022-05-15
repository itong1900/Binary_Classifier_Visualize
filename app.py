from debugpy import configure
import streamlit as st

import numpy as np
from numpy import pi
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from libraries.binary_classifier_tool import binary_classifier_tool

from libraries.models import knn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.utils import class_weight

# from keras.models import Sequential
# from keras.layers import Dense
# import tensorflow as tf


def main():
    st.set_page_config(layout="wide")
    st.sidebar.header("Configurations")

    mode = st.sidebar.radio(
        "", ("Input Data", "Classifier", "Metrics")
    )

    if mode == "Input Data":
        st.header("Data")

        data_style = st.radio("Data Generation", ('normal', 'spiral'))
        if data_style == "normal":
        # container = st.container()
        # configure_saved = container.button('Save Config?')

            col1, col2, col3, _ = st.columns(4)

            ## Initialize or set Class A configurations
            col1.write("Class A Configurations")
            st.session_state.class_A_X1_center = col1.number_input(
                'Class A, X_1 center', 
                value = st.session_state.class_A_X1_center if 'class_A_X1_center' in st.session_state else 10)
            st.session_state.class_A_X2_center = col1.number_input(
                'Class A, X_2 center', 
                value = st.session_state.class_A_X2_center if 'class_A_X2_center' in st.session_state else 10)
            st.session_state.class_A_X1_sd = col1.number_input(
                'Class A, X_1 SD', 
                value = st.session_state.class_A_X1_sd if 'class_A_X1_sd' in st.session_state else 5)
            st.session_state.class_A_X2_sd = col1.number_input(
                'Class A, X_2 SD', 
                value = st.session_state.class_A_X2_sd if 'class_A_X2_sd' in st.session_state else 5)
            st.session_state.class_A_size = col1.number_input(
                'Class A size', 
                value = st.session_state.class_A_size if 'class_A_size' in st.session_state else 5000)
            
            ## Class A np matrix
            class_A_matrix = np.concatenate( (np.random.normal(st.session_state.class_A_X1_center, st.session_state.class_A_X1_sd, st.session_state.class_A_size).reshape(-1, 1), 
                                              np.random.normal(st.session_state.class_A_X2_center, st.session_state.class_A_X2_sd, st.session_state.class_A_size).reshape(-1, 1),
                                              np.ones((st.session_state.class_A_size, 1))),
                                              axis = 1)
        
            ## Initialize or set Class B configurations
            col2.write("Class B Configurations")
            st.session_state.class_B_X1_center = col2.number_input(
                'Class B, X_1 center', 
                value = st.session_state.class_B_X1_center if 'class_B_X1_center' in st.session_state else 0)
            st.session_state.class_B_X2_center = col2.number_input(
                'Class B, X_2 center', 
                value = st.session_state.class_B_X2_center if 'class_B_X2_center' in st.session_state else 0)
            st.session_state.class_B_X1_sd = col2.number_input(
                'Class B, X_1 SD', 
                value = st.session_state.class_B_X2_sd if 'class_B_X2_sd' in st.session_state else 3)
            st.session_state.class_B_X2_sd = col2.number_input(
                'Class B, X_2 SD', 
                value = st.session_state.class_B_X2_sd if 'class_B_X2_sd' in st.session_state else 3)
            st.session_state.class_B_size = col2.number_input(
                'Class B size', 
                value = st.session_state.class_B_size if 'class_B_size' in st.session_state else 1000)

            ## Class B np matrix
            class_B_matrix = np.concatenate( (np.random.normal(st.session_state.class_B_X1_center, st.session_state.class_B_X1_sd, st.session_state.class_B_size).reshape(-1, 1), 
                                              np.random.normal(st.session_state.class_B_X2_center, st.session_state.class_B_X2_sd, st.session_state.class_B_size).reshape(-1, 1),
                                              np.zeros((st.session_state.class_B_size, 1))),
                                              axis = 1)

            st.session_state.raw_data = np.append(class_A_matrix, class_B_matrix, axis=0)
            np.random.shuffle(st.session_state.raw_data)

            df = pd.DataFrame(data=st.session_state.raw_data, columns=["X_1", "X_2", "Class"])
            df["Class"] = df["Class"].astype("category")
    
            fig = px.scatter(df, x="X_1", y="X_2", color="Class", width=600, height=600)
            fig.add_trace(
                go.Scatter(
                    name="axis",
                    x=[0, 0, None, min(df["X_1"]) , max(df["X_1"]) ],
                    y=[min(df["X_2"]) , max(df["X_2"]),  None, 0, 0],
                    mode="lines",
                    line=go.scatter.Line(color="purple"),
                    showlegend=False)
                )

            col3.plotly_chart(fig, use_container_width=False)
        
        elif data_style == "spiral":
            col1, col2, col3, _ = st.columns(4)

            ## Initialize or set Class A configurations
            col1.write("Class A Configurations")
            st.session_state.class_A_size = col1.number_input(
                'Class A size', 
                value = st.session_state.class_A_size if 'class_A_size' in st.session_state else 1000)
            st.session_state.class_A_theta_size = col1.number_input(
                'Class A Spiral Half Cycle', 
                value = st.session_state.class_A_theta_size if 'class_A_theta_size' in st.session_state else 2)
            st.session_state.class_A_noise = col1.number_input(
                'Class A, SD', 
                value = st.session_state.class_A_noise if 'class_A_noise' in st.session_state else 1)

            theta_1 = np.sqrt(np.random.rand(st.session_state.class_A_size))* st.session_state.class_A_theta_size * pi # np.linspace(0,2*pi,100)

            r_a = 2*theta_1 + pi
            data_a = np.array([np.cos(theta_1)*r_a, np.sin(theta_1)*r_a]).T
            x_a = data_a + st.session_state.class_A_noise * np.random.randn(st.session_state.class_A_size, 2)


            ## Class A np matrix
            class_A_matrix = np.concatenate( (x_a[:, 0].reshape(-1, 1), 
                                              x_a[:, 1].reshape(-1, 1),
                                              np.ones((st.session_state.class_A_size, 1))),
                                              axis = 1)

            ## Initialize or set Class B configurations
            col2.write("Class B Configurations")
            st.session_state.class_B_size = col2.number_input(
                'Class B size', 
                value = st.session_state.class_B_size if 'class_B_size' in st.session_state else 1000)
            st.session_state.class_B_theta_size = col2.number_input(
                'Class B Spiral Half Cycle', 
                value = st.session_state.class_B_theta_size if 'class_B_theta_size' in st.session_state else 2)
            st.session_state.class_B_noise = col2.number_input(
                'Class B, SD', 
                value = st.session_state.class_B_noise if 'class_B_noise' in st.session_state else 1)

            theta_2 = np.sqrt(np.random.rand(st.session_state.class_B_size))* st.session_state.class_B_theta_size * pi # np.linspace(0,2*pi,100)

            r_b = -2*theta_2 - pi
            data_b = np.array([np.cos(theta_2)*r_b, np.sin(theta_2)*r_b]).T
            x_b = data_b + st.session_state.class_B_noise * np.random.randn(st.session_state.class_B_size,2)

            ## Class B np matrix
            class_B_matrix = np.concatenate( (x_b[:, 0].reshape(-1, 1), 
                                              x_b[:, 1].reshape(-1, 1),
                                              np.zeros((st.session_state.class_B_size, 1))),
                                              axis = 1)

            st.session_state.raw_data = np.append(class_A_matrix, class_B_matrix, axis=0)
            np.random.shuffle(st.session_state.raw_data)
            df = pd.DataFrame(data=st.session_state.raw_data, columns=["X_1", "X_2", "Class"])
            df["Class"] = df["Class"].astype("category")

            fig = px.scatter(df, x="X_1", y="X_2", color="Class", width=600, height=600)
            fig.add_trace(
                go.Scatter(
                    name="axis",
                    x=[0, 0, None, min(df["X_1"]) , max(df["X_1"]) ],
                    y=[min(df["X_2"]) , max(df["X_2"]),  None, 0, 0],
                    mode="lines",
                    line=go.scatter.Line(color="purple"),
                    showlegend=False)
                )

            col3.plotly_chart(fig, use_container_width=False)


    elif mode == "Classifier":
        st.header("Train and test the data")

        train_size = st.slider("Pick Training Size", min_value=0.0, max_value=1.0, value=st.session_state.train_size if 'train_size' in st.session_state else 0.7)

        ## preprocess data
        #permuted_data = st.session_state.raw_data.sample(frac = 1, random_state = 100).reset_index(drop = True)
        X = st.session_state.raw_data[:, 0:2]
        y = st.session_state.raw_data[:, 2]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1 - train_size, random_state = 100) 

        classifier_name = st.selectbox("Choose Type of Classifier", ["Dumb Classifier", "Logistic Regression", "KNN Classifier", "SVM", "Neural Network", "KNN Handcraft"])
        
        if classifier_name == "Dumb Classifier":
            y_pred = dumbClassifer(X_test)
        elif classifier_name == "Logistic Regression":
            logmodel = LogisticRegression(random_state=100).fit(X_train, y_train)
            y_pred = logmodel.predict_proba(X_test)[:,1]
        elif classifier_name == "KNN Classifier":
            neigh = KNeighborsClassifier(n_neighbors=6)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict_proba(X_test)[:,1]
        elif classifier_name == "KNN Handcraft":
            knn_classifier = knn.KNearestNeighbor()
            knn_classifier.train(X_train.to_numpy(), y_train.to_numpy())
            dists = knn_classifier.compute_distances_no_loops(X_test.to_numpy())
            y_pred = knn_classifier.predict_labels(dists, k=8) 
        elif classifier_name == "SVM":
            svm_clf = svm.SVC(probability=True)
            svm_clf.fit(X_train, y_train)
            y_pred = svm_clf.predict_proba(X_test)[:, 1]
            #print(y_pred.shape)
        # elif classifier_name == "Neural Network":
        #     model = Sequential() 
        #     model.add(Dense(60, activation='relu', input_dim=2))
        #     model.add(Dense(1, activation='sigmoid')) 
        #     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) 

        #     model.fit(X_train, y_train, epochs=20, batch_size=32)
        #     y_pred = model.predict(X_test).ravel()
        #     #print(y_pred.ravel())

        model_saved = st.button('Save Model?')

        if model_saved:
            st.session_state.y_pred = y_pred
            st.session_state.y_test = y_test


    elif mode == "Metrics":
        st.header("Metrics")
        
        obj = binary_classifier_tool(st.session_state.y_pred, st.session_state.y_test, step_size = 0.001)
        obj.metrics_precompute()
        col1, col2 = st.columns(2)

        decision_threshold = col1.slider("Threshold: ", 0.0, 1.0, 0.5)
        col1.plotly_chart(obj.plot_all_in_one_graph(threshold = decision_threshold), use_container_width=True)
        
        col2.plotly_chart(obj.plot_ROC(threshold = decision_threshold), use_container_width=False)
        col2.plotly_chart(obj.plot_PR_Curve(threshold = decision_threshold), use_container_width=False)


        col3, col4 = st.columns(2)
        TP_num, FP_num, TN_num, FN_num, precision, recall, FPR, ABC, w_l, w_r = obj.get_CM_metrics(threshold=decision_threshold)

        metrics_df = pd.DataFrame({'Metrics':['Precision', 'Recall', 'FPR', 'Area Between Curves', "#TP", "#FP", "#TN", "#FN"],
                                   'Value':[precision, recall, FPR, ABC, int(TP_num), int(FP_num), int(TN_num), int(FN_num)]}) 

        col1.dataframe(metrics_df)


        #st.plotly_chart(obj.plot_histogram())


## Dumb Classifier
def dumbClassifer(feature_matrix):
    return np.random.uniform(0, 1, size = feature_matrix.shape[0])


if __name__ == '__main__':
    main()