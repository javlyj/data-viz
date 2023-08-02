import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import ipywidgets as widgets
import requests
import io
import base64
from PIL import Image
from IPython.display import display


app = dash.Dash(__name__)
train_clinical_data = pd.read_csv("C:/Users/Dave/Desktop/dataviz data/train_clinical_data.csv")
train_peptides = pd.read_csv("C:/Users/Dave/Desktop/dataviz data/train_peptides.csv")
train_protiens = pd.read_csv("C:/Users/Dave/Desktop/dataviz data/train_proteins.csv")
supplemental_clinical_data = pd.read_csv("C:/Users/Dave/Desktop/dataviz data/supplemental_clinical_data.csv")
combined = pd.concat([train_clinical_data, supplemental_clinical_data]).reset_index(drop=True)
smape_baseline = 95.76  # Replace with actual values
smape_updrs40 = 69.51
smape_additional = 69.42
smape_med = 67.61
smape_protein = 69.71

labels = ["UPDRS Part 1", "UPDRS Part 2", "UPDRS Part 3", "UPDRS Part 4"]
features = ["updrs_1", "updrs_2", "updrs_3", "updrs_4"]

model_image_links = {
    "CatBoost": "https://gcdnb.pbrd.co/images/45b6RyFfZBuz.png?o=1",
    "LightGBM": "https://gcdnb.pbrd.co/images/pAN3VUdP7Fpn.png?o=1",
    "Random Forest": "https://gcdnb.pbrd.co/images/0Ypf3YSHGXS5.png?o=1"
}

app.layout = html.Div([
    html.H1("UPDRS Graphs Dashboard"),
    
    dcc.Dropdown(
        id="updrs-dropdown",
        options=[
            {"label": label, "value": feature} for label, feature in zip(labels, features)
        ],
        value=features[0],
        multi=False
    ),
    
    dcc.Graph(id="updrs-graph"),
    
    dcc.Dropdown(
        id="model-dropdown",
        options=[
            {"label": model, "value": model} for model in model_image_links.keys()
        ],
        value=list(model_image_links.keys())[0],
        multi=False
    ),
    
    
    html.Img(id="image-display", src=""),
    dcc.Graph(id="smape-bar-plot")
    
])

@app.callback(
    Output("updrs-graph", "figure"),
    Output("image-display", "src"),
    Output("smape-bar-plot", "figure"),
    [Input("updrs-dropdown", "value"), Input("model-dropdown", "value")]
)
def update_graph_image_and_smape(selected_feature, selected_model):
    index = features.index(selected_feature)
    feature = features[index]
    
    # Update the UPDRS histogram graph
    fig = px.histogram(
        combined, 
        x=feature, 
        marginal="box",
        title=f"{labels[index]} Scores by Data Source",
        color_discrete_sequence=["blue"]
    )
    
    fig.update_layout(
        xaxis_title="Months",
        yaxis_title="Count",
        plot_bgcolor="white",
        font=dict(
            family="Arial",
            size=12,
            color="black"
        ),
        barmode="overlay",
        bargap=0.05,
        bargroupgap=0.1
    )
    
    # Update the model image display
    selected_image_link = model_image_links[selected_model]
    response = requests.get(selected_image_link)
    img = Image.open(io.BytesIO(response.content))
    
    img_byte_array = io.BytesIO()
    img.save(img_byte_array, format=img.format)
    img_base64 = base64.b64encode(img_byte_array.getvalue()).decode()
    
    src = f"data:image/{img.format.lower()};base64,{img_base64}"
    
    # Update the SMAPE bar plot
    x_values = ["Baseline", "Constant UPDRS 4", "Supplemental Data", "Medication State", "Protein Data"]
    y_values = [smape_baseline, smape_updrs40, smape_additional, smape_med, smape_protein]
    
    smape_fig = px.bar(
        x=x_values,
        y=y_values,
        title="SMAPE Score (Lower is Better) all with Catboost",
        labels={"x": "", "y": "SMAPE"},
        color_discrete_sequence=["blue"],
    )
    smape_fig.add_shape(
        type="line",
        x0=-0.5,
        x1=4.5,
        y0=69.51,
        y1=69.51,
        line=dict(color="red", dash="dash"),
    )
    smape_fig.update_yaxes(range=[61, 97])
    
    for i, y_val in enumerate(y_values):
        smape_fig.add_annotation(
            x=i,
            y=y_val,
            text="{:.4f}".format(y_val),
            showarrow=True,
            arrowhead=1,
            yshift=10,
        )
    
    return fig, src, smape_fig

if __name__ == '__main__':
    app.run_server(debug=True)