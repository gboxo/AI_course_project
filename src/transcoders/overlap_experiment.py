from tc_modules import *
import gc





if __name__ == "__main__":
    model = HookedTransformer.from_pretrained("gpt2")
    sae_dict = get_attention_sae_dict(layers = [0,1,2,3,4,5])
    saes = [value for value in sae_dict.values()]


    with open("5-att-kk-148.json", "r", encoding = "utf-8") as f:
        feat_dict = json.load(f)

    strings = [model.to_tokens(elem["tokens"], prepend_bos = False).reshape(-1)  for i,elem in enumerate(feat_dict["activations"]) if i%2 == 0]
    max_pos = [elem["maxValueTokenIndex"] for i,elem in enumerate(feat_dict["activations"]) if i%2 == 0]

    max_vals = [elem["maxValue"] for i,elem in enumerate(feat_dict["activations"]) if i%2 == 0]


    transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"

    tcs_dict = {}
    for i in range(5):
        tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
        tcs_dict[tc.cfg.hook_point] = tc
    tcs = list(tcs_dict.values())

    all_nodes = [] 
    for string,pos,_ in zip(strings, max_pos, max_vals):
        candidates = None
        with apply_tc(model, tcs):
            with apply_sae(model, saes):
                with model.hooks([(f"blocks.{i}.attn.hook_attn_scores", detach_hook) for i in range(12)]):

                    attributor = HierachicalAttributor(model = model)

                    target = Node("blocks.5.attn.hook_z.sae.hook_sae_acts_post",reduction=f"0.{pos+1}.148")
                    if candidates is None:

                        candidates = [Node(f"{sae.cfg.hook_name}.sae.hook_sae_acts_post") for sae in saes[:-1]] + [Node(f"{sae.cfg.out_hook_point}.sae.hook_hidden_post") for sae in tcs] + [Node(f"blocks.{i}.attn.hook_attn_scores") for i in range(12)]
                    circuit = attributor.attribute(toks=string, target=target, candidates=candidates, threshold=0.1)
                    gc.collect()
        all_nodes.append(circuit)
        del circuit, attributor


# Measure the overlap between the circuits and the top appearng nodes
from collections import Counter
# Compute top appearing nodes
def get_top_appearing_nodes(all_nodes, n=100):
    all_leaf_nodes = []
    for nodes in all_nodes:
        #Handle situation where get_leaf_nodes returns None 
        leaf_nodes = get_leaf_nodes(nodes)
        if leaf_nodes is not None:
            all_leaf_nodes.extend(leaf_nodes)

    counter = Counter(all_leaf_nodes)
    return counter.most_common(n)


def overlap(top, circuit):
    

    leaf_nodes = get_leaf_nodes(circuit)
    if leaf_nodes is None:
        return 0
    print(leaf_nodes)
    return len(set(top) & set(leaf_nodes)) / len(set(top))

def get_leaf_nodes(circuit):
    leaf_nodes = [node.hook_point + node.reduction.split(".")[-1] for node in circuit["nodes"] if "attn_scores" not in node.hook_point and node != target]
    return leaf_nodes

def compute_overlap(all_nodes):
    overlaps = []
    top_nodes = get_top_appearing_nodes(all_nodes, 100)
    top_nodes = [node[0] for node in top_nodes]
    print(top_nodes)
    for j in range(len(all_nodes)):
        
        overlaps.append(overlap(top_nodes, all_nodes[j]))
    return overlaps


compute_overlap(all_nodes)


