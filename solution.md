## Your embedding is exactly 3 numbers per subject

Yes it is correct and same is the thing done in paper we can see Eq 17 and 18 

What we can do is concatinate 90 features of PCC and 3 Features of ALFF in order to make the features wide but what PCC is doing here is to make embedding while adding ALFF based on the weights.

## BETA Factor and Attetion Fusion Formula 

what paper says is lamda = Beta (Gamma , 1-Gamma) 

This thing is already implemed what we have to do is to change two things as while the only missing thing is the small bit of if/else logic in GraSTIACL.py that decides which combination of those two settings a given --gamma_mode value maps to

beta_convention: literal
gamma_orig_mode: signal strength 

## L_CE Not Used and Computed Wrongly

Mi_j calculated in paper with raw ALFF whil in this is gone with src and des embeddigs related Eq  13 and 14 

Moreover it is not used: 
 it is passed but not read. grad.zero() wipes it out without being read we have to add view_optimizer.step() before wiping it 

## 415k params are computed 4 times but not used

mu_logits computed in MainModel(GInfoMinMax) not used in forward pass and not used in loss calculation. It is computed 4 times in each forward pass but not used anywhere. We have to remove it from the code.

The Solution is to just write dyn_wights = none.

logits only used in view_learner to calculated gate and further gamma for it.

gate_inputs = (gate_inputs_ + edge_logits_vl) / temperature

## Beta Factor 

There we have two beta factors in CODE but 1 in paper.

In paper it is about beta convetion used to calculate lamda using gamma Beta (Gamma , 1-Gamma) 

But the beta given in GraSTICAL as BETA=0.5 and passed into enocder and futher is not used in CODE nor discussed in Paper 