import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import plotly.express as px
import pandas as pd
import seaborn as sns
import json
import torch
import numpy as np
import torch.nn as nn
from jaxtyping import Int, Float 
from datasets import load_dataset
from torch.utils.data import DataLoader, dataset
from tqdm import tqdm
from collections import defaultdict
from sae_utils import get_attention_sae_dict
from transformer_lens.ActivationCache import ActivationCache
from sae_utils import get_attention_sae_dict
import base64





feat_sims = torch.load("feat_sims.pt")
feats = feat_sims["feats"].tolist()

dist_comp = torch.load("dist_comp.pt")
top_components = torch.load("top_components.pt")

feat_pairwise = torch.load("feat_pairwise_dist_comp.pt")
comps = list(feat_pairwise.keys())
avg_dist = torch.stack([val for val in feat_pairwise.values()]).mean(dim = 0)
with open("explanations_dict.json") as f:
    explanations = json.load(f)
comp_traces = torch.load("comp_traces.pt")
#comp_traces = {k: comp_traces[k] for k in dist_comp.keys()}
feats = list(comp_traces.keys())
# This must be interchanged with Max Act ASAP

# Filter for present features
bool_array = np.array([f in feats for f in feat_sims["feats"]])
feat_sims["enc"] = feat_sims["enc"][bool_array][:,bool_array]
feat_sims["dec"] = feat_sims["dec"][bool_array][:,bool_array]
feat_sims["feats"] = np.array(feat_sims["feats"])[bool_array].tolist()


total_attrb = defaultdict(dict) 

for feat,trace in comp_traces.items():
    for ex_idx,ex_trace in trace.items():
        total_attrb[feat][ex_idx] = ex_trace["target_act"]

total_attrb_per_comp = defaultdict(dict) 
for feat,trace in comp_traces.items():
    for ex_idx,ex_trace in trace.items():
        ex_trace = ex_trace["Mean trace"]
        for comp,tr in ex_trace.items():
            if not total_attrb_per_comp[feat].get(comp):
                total_attrb_per_comp[feat][comp] = []
                total_attrb_per_comp[feat][comp].append(tr.detach().sum()) 
            else:
                total_attrb_per_comp[feat][comp].append(tr.detach().sum()) 
# Convert to tensor
for feat,trace in total_attrb_per_comp.items():
    for comp,tr in trace.items():
        total_attrb_per_comp[feat][comp] = torch.stack(tr)





# Sample data for the scatter plot

# Title of the app
st.title("AI Alignment Course Final Project")

# Page selection menu
page = st.selectbox("Select a page:", ["Feature Exploration", "Theory", "Comparisons"])

# Home Page
if page == "Feature Exploration":
    # Side panel for selection
    option = st.sidebar.selectbox("Select an option:", [f"Feature {feature}" for feature in dist_comp.keys()])
    feature = int(option.split(" ")[-1])
    dist = dist_comp[feature]

    # Compute the average distance between traes across components
    dist_list = []
    for key in dist.keys():
        dist_list.append(dist[key])
    all_dists = torch.stack(dist_list)
    average_dist = all_dists.mean(dim = 0) 
    mean_average_dist = average_dist.mean(dim = 0)
    total_feat_attrb = total_attrb[feature]
    total_feat_attrb = torch.tensor([val for val in total_feat_attrb.values()])
    x = torch.stack((mean_average_dist,total_feat_attrb),dim = 0).numpy()
    df = pd.DataFrame(x.T, columns=["Trace Similarity","Max Activation"])
    top_features = top_components[feature]
    columns = list(top_features.keys())
    columns.sort()
    columns_alias = []
    for col in columns:
        layer = col.split(".")[1]
        comp = col.split(".")[2]
        if comp == "hook_attn_out":
            comp = "SAE-Attn"
        elif comp == "hook_mlp_out":
            comp = "TC-MLP"
        columns_alias.append(f"L{layer} {comp}")
    arrays = torch.stack([val for val in top_features.values()],dim = -1).numpy()#[0]
    top_features_df = pd.DataFrame(arrays, columns=columns_alias)
    top_features_df[top_features_df == -1] = " " 
    distance_tensors = {key:val for key,val in zip(columns_alias,dist_list)}
    total_feat_attrb_per_comp = {key:val.numpy() for key,val in zip(columns_alias,[total_attrb_per_comp[feature][comp] for comp in columns])}





    

    # Explanation section
    st.header("Section 1: Explanation")
    st.write("Explanation for the selected feature from Neuronpedia, description of the feature ussing GPT4-O mini")


    st.markdown(
        """
        <style>
        .highlight {
            background-color: lightblue; /* Set the background color here */
            padding: 10px;              /* Add some padding */
            border-radius: 5px;         /* Optional: round the corners */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Display text with the light blue background
    st.markdown(f'<div class="highlight">{explanations[str(feature)]}</div>', unsafe_allow_html=True)
    # Scatter Plot section
    st.header("Section 2: Scatter Plot")
    st.write("Scatter plot of the Trace Similarity vs. Max Activation")
    scatter_fig = px.scatter(df, x="Max Activation",y="Trace Similarity",  title='Trace Similarity vs. Max Activation')
    st.plotly_chart(scatter_fig)

    # Top Features section
    st.header("Section 3: Top Features")
    st.table(top_features_df)


    # Heatmap section
    st.header("Section 4: Heatmap")
    st.write("Heatmap of the pairwise similarity between the average computational trace of each example.")

    selected_tensor_name = st.selectbox("Select a Distance Matrix:", list(distance_tensors.keys()))

    # Retrieve the selected tensor
    selected_tensor = distance_tensors[selected_tensor_name]

    df = pd.DataFrame(selected_tensor.numpy(), columns=range(selected_tensor.shape[0]))

    st.subheader(f"Heatmap for {selected_tensor_name}")
    custom_colorscale = 'Plasma'  # You can choose any Plotly colorscale or create a custom one

# Create the heatmap
    heatmap = go.Figure(data=go.Heatmap(
        z=df.values,
        x=df.columns[::-1],  # Reversing the x-axis to flip horizontally (if needed)
        y=df.index[::-1],    # Reversing the y-axis to flip vertically (if needed)
        colorscale=custom_colorscale,  # Use the custom colorscale
        colorbar=dict(title='Scale'),
    ))


    heatmap.update_layout(
        title=f'Heatmap of {selected_tensor_name}',
        xaxis_title='Example',
        yaxis_title='Example',
        width=800, 
        height=400,
    )

# Display the Plotly figure in Streamlit
    st.plotly_chart(heatmap)

    # Line plot section
    st.header("Section 5: Line Plot")
    st.write("Line plot of the running mean of the average pairwise similarity between the traces of each example.")
    # Create a Plotly figure
    fig = go.Figure()

    # Add each line to the figure
    
    inverted_dict = {}

# Number of original tensors (in this case, it's 3)
    num_tensors = len(total_feat_attrb_per_comp)

# Inverting the dictionary
    for index in range(len(next(iter(total_feat_attrb_per_comp.values())))):  # Length of the tensors
        inverted_dict[index] = np.array([tensor[index] for tensor in total_feat_attrb_per_comp.values()])






    for key, tensor in inverted_dict.items():
        fig.add_trace(go.Scatter(
            x=list(range(len(tensor))),  # X-axis values
            y=tensor,              # Y-axis values (tensor)
            mode='lines+markers',   # Display lines with markers
            name=key               # Name of the line (used in the legend)
        ))

    # Update layout for better appearance
    fig.update_layout(
        title="Total Attribution per Component",
        xaxis_title="Components",
        yaxis_title="Values",
        xaxis=dict(tickmode='array', tickvals=list(range(len(total_feat_attrb_per_comp))), ticktext=list(total_feat_attrb_per_comp.keys())),
        height=600
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig)




    






    # Comparison section
    st.header("Section 6: Comparison")
    st.write("Here is a looping GIF for comparison:")

    gif_path = f"graph_evolution/graph_evolution_feat_{feature}.gif"  # Replace with your GIF file path
    file_ = open(gif_path, "rb")
    contents = file_.read()
    data_url = base64.b64encode(contents).decode("utf-8")
    file_.close()

    st.markdown(
        f'<img src="data:image/gif;base64,{data_url}" alt="Graph gif">',
        unsafe_allow_html=True,
    )
elif page == "Theory":
    st.header("Theory")
    
    st.subheader("Summary")
    st.markdown("""
    Since the introduction of dictionary learning techniques to the field of Mechanistic Interpretability (MI), significant progress has been made in understanding the computations performed by transformer models. This is done through interpretable units (Features) and their composition (Circuits).

    This project builds upon prior research on feature attribution techniques that leverage dictionary learning to gain a detailed understanding of the key components driving upstream activations.

    In this project, we:
    - Explore Attention Features from Layer 5 in GPT-2 small.
    - Apply both Sparse Autoencoders and Transcoders to create a linear circuit (following **Cite Linear Circuits**).
    - Use **Hierarchical Attribution** to prune the computational graph in the feature basis.
    - Use the concept of **Computational Trace** to generate a fingerprint of the computation that activates a given feature.
    - Investigate the relationship between the Computational Trace and the Feature that is activated.
    """)

    st.subheader("Sparse Autoencoders and Transcoders")
    st.markdown("""
    **Sparse Autoencoders**

    Sparse Autoencoders (SAEs) are a well-known technique in Mechanistic Interpretability introduced in 2022. They decompose activations at specific locations (e.g., the Residual Stream, Attention Output, or MLP Output) into Monosemantic and Interpretable Features.

    This is achieved by training an overcomplete Autoencoder to reconstruct the activations while enforcing sparsity in the latent representations. Various architectures have been proposed, including JumpRelu and TopK.

    **Transcoders**

    Recently, a variation of the SAE paradigm called the Transcoder (or Skip SAE) has been proposed. Instead of reconstructing activations, the Transcoder computes the output of a given component, typically the MLP layer of the transformer, based on its input.

    Although the objective of the Transcoder differs from that of the SAE, the sparsity constraint and the overcompleteness of the latent space are still enforced to produce monosemantic and interpretable features.
    """)

    st.subheader("Hierarchical Attribution")
    st.markdown("""
    **Gradient Attribution**

    Gradient attribution has long been a popular technique in Explainable AI (XAI) for identifying which layers, neurons, or inputs are most important for a given output or neuron activation.

    While gradient attribution is a powerful method, it has well-documented limitations, such as a lack of interpretability and robustness to examples.

    In this project, we address some of these limitations using two approaches:

    1) Perform attribution in the feature basis following recent work [^1] [^2] [^3].
    2) Use **Hierarchical Attribution** [^1], a technique that detaches unimportant nodes in the computational graph during backpropagation.

    **Hierarchical Attribution**

    Hierarchical Attribution is a technique that prunes unimportant nodes during backpropagation. It does so by computing the importance of each node in the computational graph and detaching nodes below a certain threshold.

    Paired with Transcoders and Attention Pattern Freezing, this method enables us to obtain a linear circuit that attributes an upstream component of the computational graph (not necessarily the output).
    """)

    st.subheader("Computational Trace")
    st.markdown("""
    **Computational Trace**

    The concept of Computational Trace is used throughout this project to refer to an aggregation of the features most important for the activation of a given feature in context.

    A simple Computational Trace could be the sum of attributions over the last _n_ positions for each layer/location.

    More elaborate schemes can be used, such as restricting the sum to the _n_ positions that contribute 80% of the total attribution.
    """)

    st.subheader("Citations")
    st.markdown("""
    [^1] Ge, Xuyang, et al. "Automatically Identifying Local and Global Circuits with Linear Computation Graphs." arXiv preprint arXiv:2405.13868 (2024).
    
    [^2] Marks, Samuel, et al. "Sparse Feature Circuits: Discovering and Editing Interpretable Causal Graphs in Language Models." arXiv preprint arXiv:2403.19647 (2024).
    
    [^3] Dunefsky, Jacob, Philippe Chlenski, and Neel Nanda. "Transcoders Find Interpretable LLM Feature Circuits." arXiv preprint arXiv:2406.11944 (2024).
    """)
# Comparisons Page
elif page == "Comparisons":
    # Display the scatter plot
    st.header("Encoder Similarity vs Trace Similarity")
    st.write("Scatter plot of the similarity between the encoder of two features vs. the similarity between the computational trace of the same two features averaged over the examples.")
    x = torch.stack((avg_dist.triu(1).reshape(-1),feat_sims["enc"].triu(1).reshape(-1)),dim = 0).detach().numpy()
    df = pd.DataFrame(x.T, columns=["Trace Similarity","Encoder Similarity"])
    scatter_fig = px.scatter(df, x="Trace Similarity", y="Encoder Similarity", title='Encoder Similarity vs Trace Similarity')
    st.plotly_chart(scatter_fig)




    st.header("Decoder Similarity vs Trace Similarity")
    st.write("Scatter plot of the similarity between the decoder of two features vs. the similarity between the computational trace of the same two features averaged over the examples.")
    x = torch.stack((avg_dist.triu(1).reshape(-1),feat_sims["dec"].triu(1).reshape(-1)),dim = 0).detach().numpy()
    df = pd.DataFrame(x.T, columns=["Trace Similarity","Decoder Similarity"])
    scatter_fig = px.scatter(df, x="Trace Similarity", y="Decoder Similarity", title='Decoder Similarity vs Trace Similarity')
    st.plotly_chart(scatter_fig)


    st.header("Graph")
    st.write("Graph of the every example trace pairwise similarity, colorized by the class they belong to.")



