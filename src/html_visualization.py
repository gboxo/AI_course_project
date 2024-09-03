

import json
import html
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
def loss_to_color(loss: float) -> str:
    """Convert a loss value to a color intensity."""
    # Normalize the loss to a range of 0-1 for color mapping
    normalized_loss = min(max(loss, 0), 1)  # Ensure loss is between 0 and 1
    # Use a gradient from green (low loss) to red (high loss)
    red = int(255 * normalized_loss)
    green = 255 - red
    return f'rgba({red}, {green}, 0, 1)'  # RGBA format





def create_visualization_loss(model,tokens_list, tokens_loss, is_active):

    html_content = """
    <html>
    <head>
    <style>
        .token {font-size: 14px; margin: 2px; padding: 2px; display: inline-block;}
        .prompt-group {margin-bottom: 20px;}
        .prompt-title {font-weight: bold; margin-bottom: 10px;}
    </style>
    </head>
    <body>
    <h1>Token Prediction Visualization</h1>
    """

    for i, tok in tqdm(enumerate(tokens_list)):
        html_content += f'<div class="prompt-title">Prompt {i}</div>'
        
        indices = torch.zeros(len(tok), dtype=torch.float32)
        str_tokens = [model.to_string(ind) for ind in tok[0]]
        losses = tokens_loss[i][0]

        # Simulate loss values for tokens (In practice, you'd compute this)
        # Assuming loss is a float value per token, replace this with actual loss computation
        for token, loss, highlight in zip(str_tokens[1:], losses[1:], is_active[i]):
                token_text = html.escape(token)
                color = loss_to_color(loss)  # Get color based on loss
                border_style = "border: 3px solid yellow;" if highlight else ""  # Apply border if highlight is True
                html_content += f'<span class="token" style="background-color: {color}; {border_style}">{token_text}</span> '
            
        html_content += "<div><strong>Original Text:</strong> "
        html_content += html.escape(model.to_string(tok[0, 1:])).replace(' ', '&nbsp;')
        html_content += '</div><br>'
    
    html_content += '</body></html>'

    with open("vis0.html", "w") as f:
        f.write(html_content)











def create_visualization(token_dataset, final_dict,batch_size = 3):
    with open("final_dict.json", "r") as f:
        final_dict = json.load(f)

    dataset = DataLoader(token_dataset, batch_size = batch_size)


    html_content = "<html><head><style>\
        .token {font-size: 14px; margin: 2px; padding: 2px;}\
        .correct {background-color: #d4edda; border: 1px solid #c3e6cb;}\
        .incorrect {background-color: #FFFFFF; border: 1px solid #f5c6cb;}\
        .prompt-group {margin-bottom: 20px;}\
        .prompt-title {font-weight: bold; margin-bottom: 10px;}\
        </style></head><body>\
        <h1>Token Prediction Visualization</h1>"
    

    for i,d in tqdm(enumerate(dataset)):# Select the batch
        input_ids = d['tokens']
        if f"Batch {i}" not in final_dict:
            break
        batch_dict = final_dict[f"Batch {i}"]
        html_content += f'<div class="prompt-title">Batch {i}</div>'
        for doc, seq in enumerate(input_ids):
            doc_list = batch_dict[str(doc)]
            if len(doc_list) == 0:
                continue
            indices = torch.zeros(len(seq))
            str_tokens = [model.to_string(ind) for ind in seq]
            for tup in doc_list:
                for i in range(tup[0],tup[1]+1):
                    indices[i] = 1

            html_content += f'<div class="prompt-title">Document {doc}</div>'
            for token, correct in zip(str_tokens[1:], indices[1:]):
                token_text = html.escape(token)
                token_class = 'correct' if correct==1 else 'incorrect'
                html_content += f'<span class="token {token_class}">{token_text}</span> '
            html_content += "Original Text:"
            html_content += html.escape(model.to_string(seq[1:])).replace(' ', '&nbsp;')
            html_content += '</div><br>'
        html_content += '</div>'
        html_content += '</div>'
    
    html_content += '</body></html>'

    with open("vis0.html","w") as f:
        f.write(html_content)
