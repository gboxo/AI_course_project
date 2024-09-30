import torch
import pandas as pd
import json
import numpy as np
from collections import defaultdict

# Load the necessary data from .pt files
feat_sims = torch.load("feat_sims.pt")
dist_comp = torch.load("dist_comp.pt")
top_components = torch.load("top_components.pt")
feat_pairwise = torch.load("feat_pairwise_dist_comp.pt")
comp_traces = torch.load("comp_traces.pt")

# Load the explanations from the JSON file
with open("explanations_dict.json") as f:
    explanations = json.load(f)

# Convert feature data into lists
feats = feat_sims["feats"].tolist()

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



# Create a DataFrame to store the necessary data for each feature
data = []

for feature in dist_comp.keys():
    # Get the distance for the feature
    dist = dist_comp[feature]

    # Compute the average distance between traces across components
    dist_list = []
    for key in dist.keys():
        dist_list.append(dist[key])
    all_dists = torch.stack(dist_list)
    average_dist = all_dists.mean(dim=0)
    mean_average_dist = average_dist.mean(dim=0)

    # Get the total feature attribution
    total_feat_attrb = defaultdict(dict)
    for ex_idx, ex_trace in comp_traces[feature].items():
        total_feat_attrb[ex_idx] = ex_trace["target_act"]

    total_feat_attrb = torch.tensor([val for val in total_feat_attrb.values()])

    # Get explanations for the feature
    explanation = explanations[str(feature)]

    # Add to DataFrame
    data.append({
        "feature": feature,
        "mean_average_dist": mean_average_dist.item(),
        "max_activation": total_feat_attrb.numpy().mean(),  # Assuming you want the mean of activations
        "explanation": explanation
    })

# Create a DataFrame and store it
df = pd.DataFrame(data)

# Save to CSV or Pickle
df.to_csv("feature_data.csv", index=False)
