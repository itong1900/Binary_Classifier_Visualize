import pandas as pd
import numpy as np
import math

import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots

from sklearn.metrics import roc_curve
from sklearn.metrics import precision_recall_curve

class binary_classifier_tool:
    def __init__(self, y_pred, y_ground_truth, step_size = 0.01):
        self.y_pred = y_pred
        self.y_ground_truth = y_ground_truth
        self.step_size = step_size

    def metrics_precompute(self):
        self.auxilary_df = pd.DataFrame({"y_pred": self.y_pred,
                                         "y_ground_truth": self.y_ground_truth})

        bin_bounds = np.arange(0, 1.00001, self.step_size)
        #cdf_cutoff = math.floor(threshold/self.step_size)

        self.class_0_cdf_from_bottom = []
        self.class_0_cdf_from_top = []
    
        ## Get the class_0_ratio cdf from bottom
        for upper_bound in bin_bounds:

            num_class_0_bin = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "0") & (self.auxilary_df["y_pred"] < upper_bound)].shape[0]
            total_num_bin = self.auxilary_df[(self.auxilary_df["y_pred"] < upper_bound)].shape[0]

            if total_num_bin == 0:
                ratio_class_0_bin = 1
            else:
                ratio_class_0_bin = num_class_0_bin / total_num_bin 
            self.class_0_cdf_from_bottom.append(ratio_class_0_bin)

        ## Get the class_0_ratio cdf greater than threshold value
        for lower_bound in bin_bounds:

            num_class_0_bin = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "0") & (self.auxilary_df["y_pred"] > lower_bound)].shape[0]
            total_num_bin = self.auxilary_df[(self.auxilary_df["y_pred"] > lower_bound)].shape[0]

            if total_num_bin == 0:
                ratio_class_0_bin = 0
            else:
                ratio_class_0_bin = num_class_0_bin / total_num_bin 
            self.class_0_cdf_from_top.append(ratio_class_0_bin)


    def plot_all_in_one_graph(self, threshold = 0.5):
        fig = make_subplots(rows=2, cols=1,
                    row_heights=[0.2, 0.8],
                    vertical_spacing = 0.02,
                    shared_yaxes=False,
                    shared_xaxes=True)

        fig.add_trace(go.Histogram(x=self.auxilary_df[self.auxilary_df["y_ground_truth"] == "1"]['y_pred'], xbins=dict(size=0.02), name="class1"), row = 1, col = 1)
        fig.add_trace(go.Histogram(x=self.auxilary_df[self.auxilary_df["y_ground_truth"] == "0"]['y_pred'], xbins=dict(size=0.02), name="class0"), row = 1, col = 1)
        fig.add_trace(go.Scatter(name="Threshold", x=[threshold, threshold], y=[-20, 0], mode="lines", line=go.scatter.Line(color="red"), showlegend=False), row = 1, col = 1)

        cdf_cutoff = math.floor(threshold/self.step_size)
        fig.add_trace(
            go.Scatter(
                name="trend_from_lower",
                x=np.arange(0, threshold, self.step_size),
                y=self.class_0_cdf_from_bottom[:cdf_cutoff],
                mode="lines",
                line={'color': 'blue'},
                showlegend=False),
                row = 2, col = 1
            )   
        #fig = px.line(y=self.class_0_cdf_from_bottom[:cdf_cutoff], x = np.arange(0, threshold, self.step_size),  width=400, height=400)
        fig.add_trace(
            go.Scatter(
                name="trend_from_lower",
                x=np.arange(threshold, 1.000001, self.step_size),
                y=self.class_0_cdf_from_bottom[cdf_cutoff:],
                mode="lines",
                line={'color': 'grey'},
                showlegend=False),
                row = 2, col = 1
            )
        fig.add_trace(
            go.Scatter(
                name="upper half",
                x=np.arange(threshold, 1.00001, self.step_size),
                y=self.class_0_cdf_from_top[cdf_cutoff: ],
                mode="lines",
                line={'color': 'blue'},
                showlegend=False),
                row = 2, col = 1
            )
        fig.add_trace(
            go.Scatter(
                name="trend_from_upper",
                x=np.arange(0, threshold, self.step_size),
                y=self.class_0_cdf_from_top[:cdf_cutoff],
                mode="lines",
                line={'color': 'grey'},
                showlegend=False),
                row = 2, col = 1
            )
        
        fig.add_trace(
            go.Scatter(
                name="Boundaries",
                x=[1, 1, None, -0.1, 1.1, None, 0, 0, None, 0, 1],
                y=[0, 1, None, 0, 0, None, -0.15, 1.15, None, 1, 1],
                mode="lines",
                line=go.scatter.Line(color="purple"),
                showlegend=False),
                row = 2, col = 1
            )

        fig.add_trace(
            go.Scatter(
                name="Threshold",
                x=[threshold, threshold],
                y=[-0.1, 1.1],
                mode="lines",
                line=go.scatter.Line(color="red"),
                showlegend=False),
                row = 2, col = 1
            )

        baseline = sum(self.y_ground_truth == "0")/len(self.y_ground_truth)
        fig.add_trace(
            go.Scatter(
                name="Baseline",
                x=[-0.05, 1.05],
                y=[baseline, baseline],
                mode="lines",
                line={'dash': 'dot', 'color': 'green'},
                showlegend=False),
                row = 2, col = 1
            )
        
        fig.add_trace(
            go.Scatter(
                name="octant boundaries",
                x=[0, threshold, None, threshold, 1],
                y=[self.class_0_cdf_from_bottom[cdf_cutoff], self.class_0_cdf_from_bottom[cdf_cutoff], None, self.class_0_cdf_from_top[cdf_cutoff], self.class_0_cdf_from_top[cdf_cutoff]],
                mode="lines",
                line={'dash': 'dash', 'color': 'orange'},
                showlegend=False),
                row = 2, col = 1
            )
        fig.update_layout(
                bargap=0,
                bargroupgap = 0,
                width=600,
                height=650,
                legend=dict(orientation="v", yanchor="top", y = 0.98, xanchor="left", x= 0.99, font=dict(family="Courier",size=8,color="black"))
            )

        fig.update_xaxes(title_text='y_pred', row = 2, col = 1)
        fig.update_yaxes(title_text='Proportion of negative class (labeled as 0)', row = 2, col = 1)
        return fig

    def plot_ROC(self, threshold):
        fpr, tpr, _ = roc_curve(np.array(self.y_ground_truth == '1'), self.y_pred)

        fig1 = px.line(y=tpr, x = fpr)
        fig1.add_trace(
            go.Scatter(
                x=[0,1],
                y=[0,1],
                mode="lines",
                line={'dash': 'dot', 'color': 'grey'},
                showlegend=False)
            )

        _, _, _, _, _, recall_TPR, FPR, _, _, _ = self.get_CM_metrics(threshold)

        fig2 = go.Figure(data=go.Scatter(
            x=[FPR],
            y=[recall_TPR], 
            name="Cutoff",
            mode = "markers",
            marker=dict(color=[2])
        ))
        
        fig3 = go.Figure(data=fig1.data + fig2.data)

        fig3.update_layout(
                title={
                        'text': "ROC Curve",
                        'yanchor': 'top'},
                width=400,
                height=400
                )
        fig3.update_xaxes(title_text='False Positive Rate')
        fig3.update_yaxes(title_text='True Positive Rate(Recall)')
        return fig3

    def plot_PR_Curve(self, threshold):
        precision, recall, _ = precision_recall_curve(np.array(self.y_ground_truth == '1'), self.y_pred)
        fig1 = px.line(y=precision, x = recall, title="PR curve")
        baseline = sum(self.y_ground_truth == "1")/len(self.y_ground_truth)
        fig1.add_trace(
            go.Scatter(
                x=[0,1],
                y=[baseline,baseline],
                mode="lines",
                line={'dash': 'dot', 'color': 'grey'},
                showlegend=False)
            )

        _, _, _, _, Precision, recall_TPR, _, _, _, _ = self.get_CM_metrics(threshold)

        fig2 = go.Figure(data=go.Scatter(
            x=[recall_TPR],
            y=[Precision], 
            name="Cutoff",
            mode = "markers",
            marker=dict(color=[2], size = 8)
        ))

        fig3 = go.Figure(data=fig1.data + fig2.data)
        
        fig3.update_layout(
                title={
                    'text': "PR Curve",
                    'yanchor': 'top'},
                width=400,
                height=400)
        fig3.update_xaxes(title_text='Recall')
        fig3.update_yaxes(title_text='Precision')
        return fig3


    def plot_histogram(self):
        fig = px.histogram(self.auxilary_df, x="y_pred")
        return fig

    def get_CM_metrics(self, threshold = 0.5):
        TP_num = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "1") & (self.auxilary_df["y_pred"] > threshold)].shape[0]
        FP_num = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "0") & (self.auxilary_df["y_pred"] > threshold)].shape[0]
        TN_num = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "0") & (self.auxilary_df["y_pred"] <= threshold)].shape[0]
        FN_num = self.auxilary_df[(self.auxilary_df["y_ground_truth"] == "1") & (self.auxilary_df["y_pred"] <= threshold)].shape[0]

        precision = TP_num/(TP_num + FP_num)
        recall_TPR = TP_num/(TP_num + FN_num)
        FPR = FP_num/(FP_num + TN_num)

        Area_between_curve = np.sum(np.subtract(self.class_0_cdf_from_bottom, self.class_0_cdf_from_top) * self.step_size )

        weight_from_left = len(self.y_pred[self.y_pred < threshold])/len(self.y_pred)
        weight_from_right = 1 - weight_from_left

        return TP_num, FP_num, TN_num, FN_num, precision, recall_TPR, FPR, Area_between_curve, weight_from_left, weight_from_right


