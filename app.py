from debugpy import configure
import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objs as go

from libraries.binary_classifier_tool import binary_classifier_tool

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import class_weight


def main():
    st.set_page_config(layout="wide")
    st.sidebar.header("Configurations")

    mode = st.sidebar.radio(
        "", ("Input Data", "Classifier", "Metrics")
    )

    if mode == "Input Data":
        st.header("Data")
        # container = st.container()
        # configure_saved = container.button('Save Config?')

        col1, col2, col3, _ = st.columns(4)

        ## Initialize or set Class 1 configurations
        col1.write("Class 1 Configurations")
        st.session_state.class_1_X1_center = col1.number_input(
            'class 1, X_1 center', 
            value = st.session_state.class_1_X1_center if 'class_1_X1_center' in st.session_state else 10)
        st.session_state.class_1_X2_center = col1.number_input(
            'class 1, X_2 center', 
            value = st.session_state.class_1_X2_center if 'class_1_X2_center' in st.session_state else 10)
        st.session_state.class_1_X1_sd = col1.number_input(
            'class 1, X_1 SD', 
            value = st.session_state.class_1_X1_sd if 'class_1_X1_sd' in st.session_state else 5)
        st.session_state.class_1_X2_sd = col1.number_input(
            'class 1, X_2 SD', 
            value = st.session_state.class_1_X2_sd if 'class_1_X2_sd' in st.session_state else 5)
        st.session_state.class_1_size = col1.number_input(
            'Class 1 size', 
            value = st.session_state.class_1_size if 'class_1_size' in st.session_state else 10000)

        df1 = pd.DataFrame({"X_1": pd.Series(np.random.normal(st.session_state.class_1_X1_center, st.session_state.class_1_X1_sd, st.session_state.class_1_size) ),
                            "X_2": pd.Series(np.random.normal(st.session_state.class_1_X2_center, st.session_state.class_1_X2_sd, st.session_state.class_1_size) ),
                            "Class": pd.Series(["1" for _ in range(st.session_state.class_1_size)])} )
        
        ## Initialize or set Class 0 configurations
        col2.write("Class 0 Configurations")
        st.session_state.class_0_X1_center = col2.number_input(
            'Class 0, X_1 center', 
            value = st.session_state.class_0_X1_center if 'class_0_X1_center' in st.session_state else 0)
        st.session_state.class_0_X2_center = col2.number_input(
            'Class 0, X_2 center', 
            value = st.session_state.class_0_X2_center if 'class_0_X2_center' in st.session_state else 0)
        st.session_state.class_0_X1_sd = col2.number_input(
            'Class 0, X_1 SD', 
            value = st.session_state.class_0_X2_sd if 'class_0_X2_sd' in st.session_state else 3)
        st.session_state.class_0_X2_sd = col2.number_input(
            'Class 0, X_2 SD', 
            value = st.session_state.class_0_X2_sd if 'class_0_X2_sd' in st.session_state else 3)
        st.session_state.class_0_size = col2.number_input(
            'Classs 0 size', 
            value = st.session_state.class_0_size if 'class_0_size' in st.session_state else 1000)
        
        df2 = pd.DataFrame({"X_1": pd.Series(np.random.normal(st.session_state.class_0_X1_center, st.session_state.class_0_X1_sd, st.session_state.class_0_size)),
                            "X_2": pd.Series(np.random.normal(st.session_state.class_0_X2_center, st.session_state.class_0_X2_sd, st.session_state.class_0_size)),
                            "Class": pd.Series(["0" for _ in range(st.session_state.class_0_size)])})

        st.session_state.raw_data = pd.concat([df1, df2]) #df1.append(df2, ignore_index = True)

        fig = px.scatter(st.session_state.raw_data, x="X_1", y="X_2", color="Class", width=600, height=600)
        fig.add_trace(
            go.Scatter(
                name="axis",
                x=[0, 0, None, min(st.session_state.raw_data["X_1"]) , max(st.session_state.raw_data["X_1"]) ],
                y=[min(st.session_state.raw_data["X_2"]) , max(st.session_state.raw_data["X_2"]),  None, 0, 0],
                mode="lines",
                line=go.scatter.Line(color="purple"),
                showlegend=False)
            )

        col3.plotly_chart(fig, use_container_width=False)


    elif mode == "Classifier":
        st.header("Train and test the data")

        train_size = st.slider("Pick Training Size", min_value=0.0, max_value=1.0, value=st.session_state.train_size if 'train_size' in st.session_state else 0.7)

        ## preprocess data
        permuted_data = st.session_state.raw_data.sample(frac = 1, random_state = 100).reset_index(drop = True)
        
        X_train, X_test, y_train, y_test = train_test_split(permuted_data.drop('Class',axis=1), permuted_data["Class"], test_size = 1-train_size, random_state = 100) 

        classifier_name = st.selectbox("Choose Type of Classifier", ["Dumb Classifier", "Logistic Regression", "KNN Classifier", "SVM", "Neural Network"])
        
        if classifier_name == "Dumb Classifier":
            y_pred = dumbClassifer(X_test)
        elif classifier_name == "Logistic Regression":
            logmodel = LogisticRegression(random_state=100).fit(X_train, y_train)
            y_pred = logmodel.predict_proba(X_test)[:,1]
        elif classifier_name == "KNN Classifier":
            neigh = KNeighborsClassifier(n_neighbors=6)
            neigh.fit(X_train, y_train)
            y_pred = neigh.predict_proba(X_test)[:,1]
            

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