import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(torch.nn.Module):
    def __init__(self, layers):
        super(Model, self).__init__()
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
class ModelTrainer:
    def __init__(self, model, criterion, optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        
    def train(self, train_loader, epochs, device=DEVICE):
        self.model.to(device)
        for epoch in range(epochs):
            for _, data in enumerate(train_loader):
                inputs, labels = data
                labels = labels.long()
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
            if epoch % 10 == 0:
                print(f'''
Epoch: {epoch} of {epochs}
    Loss:               {loss.item()}
    Train Accuracy:     {self.test(train_loader)}
''')
                
    def test(self, test_loader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total
        # print(f"Accuracy: {100 * correct / total}")