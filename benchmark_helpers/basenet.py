## benchmark net

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
 

# loss functions: 


class BenchmarkNet(nn.Module):
    """
    NonLinear Classifier
    """
 
  
    def __init__(self, input_size, hidden_layer1, hidden_layer2, output_size, dropout_coeff=None):
        # now this output size is 3, for the 2 classes + defer
        # from the paper "Who should predict?" (Mozannar), the realizable surrogate function requires
        # outputs ( tensor ): outputs of model with K+1 output heads ( without softmax )
        # so this net will not have the softmax
        super(BenchmarkNet, self).__init__()
        self.softmax = nn.Softmax(dim=1)


        #self.flatten =nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, hidden_layer1),
            nn.ReLU(),
            nn.Dropout(dropout_coeff),

            nn.Linear(hidden_layer1, hidden_layer2),
            nn.ReLU(),
            nn.Dropout(dropout_coeff),
            nn.Linear(hidden_layer2, output_size)
        )
        
    def forward(self, x):
 
        out = self.linear_relu_stack(x)
 
        return out #return logits only
    








    
   
   