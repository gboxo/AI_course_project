
# How to fint the edges between nodes

# To find the edges between attention sccore nodes and features in lower layers we:
# 1. Select the most important attention score node, get the WQ and wK matrices for that head.
# 2. Get the decoder of the features in the lower layers present in that position.
# 3. Multiply the WDi@WQ@WK@WDj
# 4. Get the total attribution and normalize it to add up  to the total attribution of the attention score node.

# %%

# Get the  the attention score nodes of the first layer (starting from 1)

def get_attention_score_nodes(circuit,layer):
    return [node for node in circuit["nodes"] if f"blocks.{layer}" in node.hook_point and "attn_scores" in node.hook_point]


# Sort the attention score nodes by attribution
def sort_nodes_by_attribution(circuit,nodes):
    return sorted(nodes,key = lambda node: circuit["nodes"][node]["attribution"],reverse=True)

# Get the WQ and WK matrices of the attention score node
def attn_matrices(model,attn_score_node):
    layer = int(attn_score_node.hook_point.split(".")[1])
    head = int(attn_score_node.reduction.split(".")[1])
    WQ = model.blocks[layer].attn.W_Q.detach()[head]
    WK = model.blocks[layer].attn.W_K.detach()[head]
    return WQ,WK

# Get the upstream nodes in a certain position
def get_upstream_nodes_pos(circuit,pos,layer):
    upstream_nodes = []
    for node in circuit["nodes"].keys():
        node_layer = int(node.hook_point.split(".")[1])
        node_pos = node.reduction.split(".")[1]

        if "sae" in node.hook_point and node_layer < layer and node_pos == pos:
            upstream_nodes.append(node)
    return upstream_nodes

# Get the decoder of a SAE node

def get_decoder(sae_node):
    assert "attn_scores" not in sae_node.hook_point
    if "hook_sae_acts_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return saes[layer].W_dec.detach()[feature]
    elif "hook_hidden_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return tcs[layer].W_dec.detach()[feature]



# Compute the multiplication of the matrices


def compute_prods(WQ,WK,decoder1,decoder2):
    return decoder1 @ WQ @ WK.T @ decoder2.T
# Compute the attributions matrix multiplication

def compute_QK_edges(circuit,layer):
    attn_score_nodes = get_attention_score_nodes(circuit,layer)
    attn_score_nodes = sort_nodes_by_attribution(circuit,attn_score_nodes)
    for attn_score_node in attn_score_nodes:
        attrb = circuit["nodes"][attn_score_node]["attribution"]
        WQ,WK = attn_matrices(model,attn_score_node)
        pos1 = attn_score_node.reduction.split(".")[2]
        pos2 = attn_score_node.reduction.split(".")[3]
        upstream_nodes_pos1 = get_upstream_nodes_pos(circuit,pos1,layer)
        upstream_nodes_pos2 = get_upstream_nodes_pos(circuit,pos2,layer)
        prods = []
        for upstream_node1 in upstream_nodes_pos1:
            decoder1 = get_decoder(upstream_node1)
            for upstream_node2 in upstream_nodes_pos2:
                decoder2 = get_decoder(upstream_node2)
                prod = compute_prods(WQ,WK,decoder1,decoder2)
                prods.append(prod)
        for i,upstream_node1 in enumerate(upstream_nodes_pos1): 
            for j,upstream_node2 in enumerate(upstream_nodes_pos2):
                prod = prods[i*len(upstream_nodes_pos2)+j]
                circuit["edges"].append((upstream_node1,upstream_node2,{"attribution":(attrb*(prod/sum(prods))).item()}))
    return circuit




def compute_all_edges(circuit):
    for layer in range(1,6):
        circuit = compute_QK_edges(circuit,layer)
    return circuit




