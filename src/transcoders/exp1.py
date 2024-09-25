from tc_modules import *



model = HookedTransformer.from_pretrained("gpt2")
sae_dict = get_attention_sae_dict(layers = [4])
saes = [value for value in sae_dict.values()]

with open("full_dataset.json", "r") as f:
    dataset = json.load(f)

transcoder_template  = "/media/workspace/gpt-2-small-transcoders/final_sparse_autoencoder_gpt2-small_blocks.{}.ln2.hook_normalized_24576"
threshold = 0

tcs_dict = {}
for i in range(4,5):
    tc = SparseAutoencoder.load_from_pretrained(f"{transcoder_template.format(i)}.pt").eval()
    tcs_dict[tc.cfg.hook_point] = tc
tcs = list(tcs_dict.values())
data = dataset["3997"]["0"][1]
pos = dataset["3997"]["0"][0]
feature_id = "3997"
toks = model.to_string(data)
feat = int(feature_id)

