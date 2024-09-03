"""
Mullti Token Circuits

What are multlitoken circuits in the context of max  activating dataset examples for a given attention features.


It's very usual that a given feature (attention or otherwise) is active in several tokens in a prompt, and that the (DFA destination token) is outside those tokens in which the feature is active.

A common approach is to either average the activations over the token positions or just use the first token in which the feature is activates.


In this work this apporach changes and the focus is draw to all the multiple tokens in which the feature is active not just the first.


To do so  we have several approaches, but from first principles the best approach seems to iterate over the token positions and for each positions search for shallow circuits, as the position index progresses inside the fewe tokens where the feature is active, we will reference circuit formations from the previous tokens.


This introduces a tradeoff because a very similar thing can be naively done by just attributing from rhs tokens, the problem is that this attribution is not faithful at all with the causal attention mask and AR nature of language models.




"""

