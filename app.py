from debugpy import configure
import streamlit as st

import numpy as np
import pandas as pd

import plotly.express as px

from libraries.binary_classifier_tool import binary_classifier_tool

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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

        col1, col2, col3, col4 = st.columns(4)
        class_A_X1_center = col1.number_input('Classs A, X_1 center', value = 1)
        class_A_X2_center = col1.number_input('Classs A, X_2 center', value = 1)
        class_A_X1_sd = col1.number_input('Classs A, X_1 SD', value = 0.5)
        class_A_X2_sd = col1.number_input('Classs A, X_2 SD', value = 0.5)
        
        class_B_X1_center = col2.number_input('Classs B, X_1 center', value = 0)
        class_B_X2_center = col2.number_input('Classs B, X_2 center', value = 0)
        class_B_X1_sd = col2.number_input('Classs B, X_1 SD', value = 0.5)
        class_B_X2_sd = col2.number_input('Classs B, X_2 SD', value = 0.5)

        class_A_size = col1.number_input('Classs A size', value = 10000)
        class_B_size = col2.number_input('Classs B size', value = 1000)

        st.session_state.X1_class_A = np.random.normal(class_A_X1_center, class_A_X1_sd, class_A_size)
        st.session_state.X2_class_A = np.random.normal(class_A_X2_center, class_A_X2_sd, class_A_size)
        st.session_state.X1_class_B = np.random.normal(class_B_X1_center, class_B_X1_sd, class_B_size)
        st.session_state.X2_class_B = np.random.normal(class_B_X2_center, class_B_X2_sd, class_B_size)


        df1 = pd.DataFrame({"X_1": pd.Series(st.session_state.X1_class_A),
                            "X_2": pd.Series(st.session_state.X2_class_A),
                            "Class": pd.Series(["1" for _ in range(class_A_size)])})
        df2 = pd.DataFrame({"X_1": pd.Series(st.session_state.X1_class_B),
                            "X_2": pd.Series(st.session_state.X2_class_B),
                            "Class": pd.Series(["0" for _ in range(class_B_size)])})

        st.session_state.raw_data = df1.append(df2, ignore_index = True)

        fig = px.scatter(st.session_state.raw_data, x="X_1", y="X_2", color="Class", symbol="Class", width=600, height=600)
        col3.plotly_chart(fig, use_container_width=False)

        #configure_saved = st.button('Save Config?')

        # if configure_saved:
            # st.session_state.raw_data = data


    elif mode == "Classifier":
        st.header("Train and test the data")

        if 'raw_data' not in st.session_state:
            st.write("save raw data config first")

        train_size = st.slider("Pick Training Size", min_value=0.0, max_value=1.0, value=0.7)

        ## preprocess data
        permuted_data = st.session_state.raw_data.sample(frac = 1, random_state = 100).reset_index(drop = True)
        
        X_train, X_test, y_train, y_test = train_test_split(permuted_data.drop('Class',axis=1), permuted_data["Class"], test_size = 1-train_size, random_state = 100) 

        classifier_name = st.selectbox("Choose Type of Classifier", ["Dumb Classifier", "Logistic Regression", "SVM", "Neural Network"])
        
        if classifier_name == "Dumb Classifier":
            y_pred = dumbClassifer(X_test)
        elif classifier_name == "Logistic Regression":
            # recommended_weights = class_weight.compute_class_weight(class_weights = 'balanced',
            #                                                         classes = np.unique(y_train),
            #                                                         y = y_train)
            # class_0_weight = st.slider("class weight on fit class 0", min_value=0.0, max_value=1.0, value=recommended_weights['0'])

            # class_weight_lgr = {"0": class_0_weight,
            #                     "1": class_0_weight * }

            logmodel = LogisticRegression(random_state=100).fit(X_train, y_train)
            y_pred = logmodel.predict_proba(X_test)[:,1]

        model_saved = st.button('Save Model?')

        if model_saved:
            st.session_state.y_pred = y_pred
            st.session_state.y_test = y_test


    elif mode == "Metrics":
        st.header("Metrics")
        decision_threshold = st.slider("Threshold: ", 0.0, 1.0, 0.5)

        obj = binary_classifier_tool(st.session_state.y_pred, st.session_state.y_test, step_size = 0.01)
        obj.metrics_precompute()
        st.plotly_chart(obj.plot_all_in_one_graph(threshold = decision_threshold), use_container_width=False)
        st.plotly_chart(obj.plot_ROC(threshold = decision_threshold), use_container_width=False)
        st.plotly_chart(obj.plot_PR_Curve(threshold = decision_threshold), use_container_width=False)


        TP_num, FP_num, TN_num, FN_num, precision, recall, FPR, ABC, w_l, w_r = obj.get_CM_metrics(threshold=decision_threshold)
        st.write("#TP = ", TP_num)
        st.write("#FP = ", FP_num)
        st.write("#TN = ", TN_num)
        st.write("#FN = ", FN_num)
        
        st.write("weight from left = ", w_l)
        st.write("weight from right = ", w_r)
        st.write("precision = ", precision)
        st.write("recall = ", recall)
        st.write("FPR = ", FPR)
        st.write("Area Between Curves = ", ABC)

        st.plotly_chart(obj.plot_histogram())


## Dumb Classifier
def dumbClassifer(feature_matrix):
    return np.random.uniform(0, 1, size = feature_matrix.shape[0])


if __name__ == '__main__':
    main()