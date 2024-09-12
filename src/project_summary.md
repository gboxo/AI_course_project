# Unsupervised Multi-Token Circuit Finding



## Introduction



Since the introduction of dictionary learning techniques to the field of MI, rapid progress has been made in the development of general techniques to understand the computations performed by a transformer model trough the lens of intepretable units (Features) and their composition (Circuits).

This project builds upon prior literature on circuit finding techniques that leverage dictionary learning and present methods and results, with a focus on sequences of tokens, where the model has a high degree of certainty in it's predictions. 



## Objective





## Methodology


1) Find subsequences of tokens in the pile where the model achieves relatively low loss.
2) Process the subsequences with dictionary learning techniques to extract computational traces in the interesting positioons.
    - Further filter the subsequences ussing the following criteria:
        - For token j in the subsequence, most of the attention is on tokens $s_0< i < j<s_n$.
        - Select the most improtant attentoion features across the whole subsequence
3) Cluster all the activations ussing k-means clustering on the computational traces, select k with a coverage metric.
4) Use hierarchical attribution to find circuits for some impportant attention feature (in layer 5) across the subsequences in the cluster.




