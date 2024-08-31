from torch.nn import Module

class Activation(Module):
    def __init__(self, activation):
        super(Activation, self).__init__()
        self.activation = activation

    def forward(self, x):
        return self.activation(x)