import numpy as np
## This is an opportunistic upper bound for the relay capacity 
## mentioned in the manuscript (i.e., relay can always perfectly decode), 
## will implement the exact relay capacity in the journal submission.

for snr in range(0,12,2):
    print(2*np.log2(1+10**(snr/10)))