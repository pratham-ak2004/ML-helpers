import torch
from sklearn.model_selection import train_test_split

class Dataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.X = torch.tensor(x).float()
        self.y = torch.tensor(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class DataLoader():
    def __init__(self,X,Y,batch_size=32, test_size=0.2, random_state=42, shuffle=True):
        self.data = X
        self.target = Y
        
        self.get_data_loader(batch_size=batch_size, test_size=test_size, random_state=random_state, shuffle=shuffle)
        
    def get_data(self):
        return self.X

    def get_target(self):
        return self.Y

    def get_data_split(self, test_size=0.2,random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(self.data, self.target, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def get_data_loader(self, batch_size=32, test_size=0.2, random_state=42, shuffle=True):
        X_train, X_test, y_train, y_test = self.get_data_split(test_size=test_size, random_state=random_state)
        
        self.train_loader = self.get_data_loader_from_data(X_train, y_train, batch_size, shuffle=shuffle)
        self.test_loader = self.get_data_loader_from_data(X_test, y_test, batch_size, shuffle=shuffle)
        
        return self.train_loader, self.test_loader
    
    def get_data_loader_from_data(self, X, y, batch_size=32,shuffle=True):
        dataset = Dataset(X, y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,shuffle=shuffle)
        return loader

    def get_loader(self,loader_name):
        if loader_name == "train":
            return self.train_loader
        elif loader_name == "test":
            return self.test_loader
        else:
            return self.train_loader, self.test_loader