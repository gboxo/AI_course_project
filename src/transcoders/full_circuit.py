from tc_modules import *
from collections import defaultdict
from plot_circuit import visualize_circuit
from plot_circuit2 import visualize_full_circuit




def get_attn_matrices(model,head,layer):
    WQ = model.blocks[layer].attn.W_Q.detach()[head]
    WK = model.blocks[layer].attn.W_K.detach()[head]
    WO = model.blocks[layer].attn.W_O.detach()[head]
    WV = model.blocks[layer].attn.W_V.detach()[head]

    return WQ,WK,WO,WV

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




def get_encoder(sae_node):
    assert "attn_scores" not in sae_node.hook_point
    if "hook_sae_acts_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return saes[layer].W_enc.detach()[feature]
    elif "hook_hidden_post" in sae_node.hook_point:
        layer = int(sae_node.hook_point.split(".")[1])
        feature = int(sae_node.reduction.split(".")[-1])
        return tcs[layer].W_enc.detach()[feature]





if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2",fold_ln=True, fold_value_biases = True)
    sae_dict = get_attention_sae_out_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]


    with open("full_dataset.json", "r") as f:
        full_dataset = json.load(f)
    pos = full_dataset["6628"]["9"][0]
    toks = full_dataset["6628"]["9"][1]
    toks = model.to_string(toks)



    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    tcs_dict = {}
    for i in range(6):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())




 # %%
    candidates = None
    all_saes = saes + tcs

    with apply_sae(model, all_saes):
        with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):
            attributor = HierachicalAttributor(model = model)

            target = Node("blocks.5.hook_mlp_out.sae.hook_hidden_post",reduction="0.6.6628")
    
            if candidates is None:
                candidates = [Node("hook_embed")]+[Node("hook_pos_embed")]+[Node(f"{sae.cfg.hook_name}.sae.hook_sae_acts_post") for sae in saes] + [Node(f"{sae.cfg.out_hook_point}.sae.hook_hidden_post") for sae in tcs] + [Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(12)]
            circuit,cache,target_act = attributor.attribute(toks=toks, target=target, candidates=candidates, threshold=0.1)
    #visualize_circuit(circuit)




    # How to visualize the full circuit
    """
    1) Ussing the attribution circuit, get:
        - The mlp and attn fetures for the first layer
        - The decoders of those features
        - The attention score nodes of the second layer 
    2) For each of the attention score nodes:
        - Get the QK matrices of the corresponding head
        - Compute the product ussing the feature decoders and the QK matrices
        - Get the maximum pair of features
        - Store the features that haven't been used
    3) Get the mlp and attn features of the upstream layers
        - Get the encoder of those features 
        - Get decoder feature of attn_out feaures in the first layer 
    4) For each attention score node in the first layer:
        - Get the OV matrix 
        - Compute the product using the decoder of attn features, the OV matrix and the encoder of the upstream features
        - Get the maximum pair of features
        - Store the features that haven't been used

    """

    custom_order = {"hook_attn_scores":0, "hook_attn_out":1,"hook_mlp_out":2}


# Nested dictionary dict(dict(list))
    nodes_dict = defaultdict(lambda: defaultdict(list))
    for node in circuit["nodes"]:
        name = node.hook_point
        if "embed" in name:
            continue
        layer = int(name.split(".")[1])
        comp = name.split(".")[2]
        if comp == "attn":
            comp = "attn_scores"
        nodes_dict[layer][comp].append(node)



    def custom_sort_key(key):
        # Return the order index from custom_order or a high number if the key is not in the order
        return custom_order.get(key, float('inf'))


    sorted_nodes_dict = {
        outer_key: dict(sorted(inner_dict.items(), key=lambda item: custom_sort_key(item[0])))
        for outer_key, inner_dict in sorted(nodes_dict.items())
    }
    full_circuit = {}
    full_circuit["nodes"] = defaultdict(dict) 
    full_circuit["edges"] = []
    max_layer = list(sorted_nodes_dict.keys())[-1]
    for att_scores_layer in list(sorted_nodes_dict.keys())[1:]:
        layer = att_scores_layer-1
        print(f"Layer {layer}")
        # Get the attention scores node layer 
        #att_scores_layer = layer + 1
        # Get the attention scores nodes
        if "attn_scores" not in sorted_nodes_dict[att_scores_layer]:
            continue
        att_scores_nodes = sorted_nodes_dict[att_scores_layer]["attn_scores"]
        attn_scores_nodes_dict = {"Head":[int(att_scores_node.reduction.split(".")[1]) for att_scores_node in att_scores_nodes],"Dest_Src":[(int(att_scores_node.reduction.split(".")[2]),int(att_scores_node.reduction.split(".")[3])) for att_scores_node in att_scores_nodes]}
        unique_heads = set(attn_scores_nodes_dict["Head"])
        unique_heads_mat_dict = defaultdict(dict)
        for head in unique_heads:
            WQ,WK,WO,WV = get_attn_matrices(model,head,att_scores_layer)
            unique_heads_mat_dict[head]["WQ"] = WQ
            unique_heads_mat_dict[head]["WK"] = WK
            unique_heads_mat_dict[head]["WO"] = WO
            unique_heads_mat_dict[head]["WV"] = WV


        # Get all the remaining upstream features
        upstream_features = []
        upstream_feature_decoders = []
        upstream_features_pos = []
        for l in range(att_scores_layer):
            comp_dict = sorted_nodes_dict[l]
            for comp, nodes in comp_dict.items():
                for node in nodes:
                    comp_name = node.hook_point
                    if "embed" in comp_name:
                        continue 
                    if "scores" in comp_name:
                        continue
                    upstream_features.append(node)
                    upstream_feature_decoders.append(get_decoder(node))
                    upstream_features_pos.append(int(node.reduction.split(".")[1]))

        all_decoders = torch.stack(upstream_feature_decoders)
        all_positions = torch.tensor(upstream_features_pos)
        # Compute the product
        for att_scores_node in att_scores_nodes:
            WQ = unique_heads_mat_dict[int(att_scores_node.reduction.split(".")[1])]["WQ"]
            WK = unique_heads_mat_dict[int(att_scores_node.reduction.split(".")[1])]["WK"]
            dest = int(att_scores_node.reduction.split(".")[2])
            src = int(att_scores_node.reduction.split(".")[3])
            mask = all_positions <= dest
            prod = all_decoders @ WQ @ WK.T @ all_decoders.T
            pair = prod.argmax()
            row_index = pair // prod.size(1)
            col_index = pair % prod.size(1)
            attrb = prod[row_index,col_index]
            # Add the 3 nodes in thefull circuit
            # Add the 2 edges in the full circuit
            full_circuit["nodes"][att_scores_node] = circuit["nodes"][att_scores_node]
            full_circuit["nodes"][upstream_features[row_index]] = circuit["nodes"][upstream_features[row_index]]
            full_circuit["nodes"][upstream_features[col_index]] = circuit["nodes"][upstream_features[col_index]]

            full_circuit["edges"].append((upstream_features[row_index],att_scores_node,{"attribution":attrb}))
            full_circuit["edges"].append((upstream_features[col_index],att_scores_node,{"attribution":attrb}))

        # Get the decoders of the attention features of this layer
        attn_nodes = sorted_nodes_dict[att_scores_layer]["hook_attn_out"]
        attn_features_decoders = [get_decoder(node) for node in attn_nodes]
        attn_features_decoders = torch.stack(attn_features_decoders)
        # Get downstream nodes
        downstream_features = []
        downstream_feature_encoders = []
        downstream_features_pos = []
        for l in range(att_scores_layer,max_layer):
            comp_dict = sorted_nodes_dict[l]
            for comp, nodes in comp_dict.items():
                for node in nodes:
                    comp_name = node.hook_point
                    if "embed" in comp_name:
                        continue 
                    if "scores" in comp_name:
                        continue
                    downstream_features.append(node)
                    downstream_feature_encoders.append(get_decoder(node))
                    downstream_features_pos.append(int(node.reduction.split(".")[1]))



            if len(downstream_features) == 0:
                continue
            all_encoders = torch.stack(downstream_feature_encoders)
            all_positions = torch.tensor(downstream_features_pos)
            # Compute the product
            for att_scores_node in att_scores_nodes:
                WO = unique_heads_mat_dict[int(att_scores_node.reduction.split(".")[1])]["WO"]
                WV = unique_heads_mat_dict[int(att_scores_node.reduction.split(".")[1])]["WV"]
                dest = int(att_scores_node.reduction.split(".")[2])
                src = int(att_scores_node.reduction.split(".")[3])
                mask = all_positions <= dest
                prod = attn_features_decoders @ WV @ WO @ all_encoders.T
                pair = prod.argmax()
                row_index = pair // prod.size(1)
                col_index = pair % prod.size(1)
                attrb = prod[row_index,col_index]
                # Add the 3 nodes in thefull circuit
                # Add the 2 edges in the full circuit
                full_circuit["nodes"][att_scores_node] = circuit["nodes"][att_scores_node]
                full_circuit["nodes"][attn_nodes[row_index]] = circuit["nodes"][attn_nodes[row_index]]
                full_circuit["nodes"][downstream_features[col_index]] = circuit["nodes"][downstream_features[col_index]]
                full_circuit["edges"].append((att_scores_node,attn_nodes[row_index],{"attribution":attrb}))
                full_circuit["edges"].append((att_scores_node,downstream_features[col_index],{"attribution":attrb}))


    visualize_full_circuit(full_circuit)






