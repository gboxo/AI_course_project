


import matplotlib.pyplot as plt

import seaborn as sns
import torch
import pandas as pd
import numpy as np

import html
from transformer_lens import HookedTransformer

def plot_histogram_loss_over_n_toks(avg_loss,sample_size = 100):
    """
    input:
        avg_loss: The average loss over n tokens for all the inputs it has shape [batch seq_len]
        sample_size: The values that we will use to not spend too much memory


    output
        
    """
    tensor = avg_loss.numpy()
    indices = np.random.choice(avg_loss.shape[0],sample_size, replace = False)
    indices2 = np.random.choice(avg_loss.shape[1],sample_size, replace = False)
    sample_data = tensor[indices,:]
    flattened_data = sample_data.flatten()
    sns.histplot(flattened_data,bins = 100,kde = True)

    plt.title('Histogram of Sampled Tensor Values')
    plt.xlim([0,20])
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.show()






from matplotlib.colors import ListedColormap
import os

# Define the path to the directory containing the checkpoint files
checkpoint_dir = "checkpoints/"

def load_checkpoints(checkpoint_dir):
    # Load all .pt files with the correct naming structure into one dataset
    checkpoints = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("corr_pred_") and file.endswith(".pt"):
            checkpoint = torch.load(os.path.join(checkpoint_dir, file))
            checkpoints.append(checkpoint)
    return checkpoints

def process_data(checkpoints):
    results = []
    for checkpoint in checkpoints:
        input_ids = checkpoint['Tokens']
        corr_pred = checkpoint['Corr Pred']
        for i in range(input_ids.size(0)):
            tokens = input_ids[i].tolist()
            correct_preds = corr_pred[i].tolist()
            max_consecutive_correct = max(
                (sum(1 for _ in g) for k, g in itertools.groupby(correct_preds) if k),
                default=0
            )
            results.append((tokens, correct_preds, max_consecutive_correct))
    return results
# %%


def create_visualization(results):
    # Create a DataFrame for easier manipulation and visualization
    df = pd.DataFrame(results, columns=['tokens', 'correct_preds', 'max_consecutive_correct'])
    
    # Group by the max_consecutive_correct to create the visualization
    grouped = df.groupby('max_consecutive_correct')
    html_content = "<html><head><style>\
        .token {font-size: 14px; margin: 2px; padding: 2px;}\
        .correct {background-color: #d4edda; border: 1px solid #c3e6cb;}\
        .incorrect {background-color: #f8d7da; border: 1px solid #f5c6cb;}\
        .prompt-group {margin-bottom: 20px;}\
        .prompt-title {font-weight: bold; margin-bottom: 10px;}\
        </style></head><body>\
        <h1>Token Prediction Visualization</h1>"
    
    for max_consec, group in grouped:
        html_content += f'<div class="prompt-group">\
            <div class="prompt-title">Prompts with max {max_consec} consecutive correct predictions</div>'
        for index, row in group.iterrows():
            html_content += '<div class="prompt">'
            for token, correct in zip(row['tokens'][1:], row['correct_preds']):
                token_text = model.to_single_str_token(token)
                token_text = html.escape(token_text)
                token_class = 'correct' if correct else 'incorrect'
                html_content += f'<span class="token {token_class}">{token_text}</span> '
            html_content += "Original Text:"
            html_content += html.escape(model.tokenizer.decode(row['tokens'][1:])).replace(' ', '&nbsp;')
            html_content += '</div><br>'
        html_content += '</div>'
    
    html_content += '</body></html>'



    with open("vis1.html","w") as f:
        f.write(html_content)

    


#if __name__ == "__main__":
if True:
    import itertools
    
    model = HookedTransformer.from_pretrained("gpt2-small")

    # Load and process checkpoints
    checkpoints = load_checkpoints(checkpoint_dir)
    results = process_data(checkpoints)
    create_visualization(results)


