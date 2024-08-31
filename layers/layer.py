import torch

class Linear(torch.nn.Module):
    def __init__(self,in_features, out_features, activation, **kwargs):
        super(Linear, self).__init__()
        
        self.layer = torch.nn.Linear(in_features, out_features, **kwargs)
        self.activation = activation
        
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x
    
class Conv2D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation, **kwargs):
        super(Conv2D, self).__init__()
        
        self.layer = torch.nn.Conv2d(in_channels, out_channels, kernel_size, **kwargs)
        self.activation = activation
        
    def forward(self, x):
        x = self.layer(x)
        x = self.activation(x)
        return x
    
class Dropout(torch.nn.Module):
    def __init__(self, p):
        super(Dropout, self).__init__()
        
        self.layer = torch.nn.Dropout(p)
        
    def forward(self, x):
        return self.layer(x)
    
class Dropout2D(torch.nn.Module):
    def __init__(self, p):
        super(Dropout2D, self).__init__()
        
        self.layer = torch.nn.Dropout2d(p)
        
    def forward(self, x):
        return self.layer(x)