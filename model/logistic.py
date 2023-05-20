import torch.nn as nn

###################################

class Logistic(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Logistic, self).__init__()
        self.input_dim = input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.apply(self.weights_init)
    
    def forward(self, x):
        out = self.linear(x.view(-1,self.input_dim))
        return out
    
    def weights_init(self,module):
        if isinstance(module, nn.Linear):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)