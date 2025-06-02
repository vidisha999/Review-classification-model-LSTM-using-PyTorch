
# Defing the Parameters:
max_features = 2000
batch_size = 50
vocab_size = max_features
from sklearn.model_selection import train_test_split
import torch


class Create:

    def create_dataset(self,X,Y):
    
        # Train, Test & validation Splits
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.10, random_state=42)
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.10, random_state=42)
        print(X_train.shape, Y_train.shape)
        print(X_test.shape, Y_test.shape)
        print(X_val.shape, Y_val.shape)
        return X_train, X_test,X_val Y_train, Y_test,Y_val

    def data_loader(self,X_train, X_val, Y_train, Y_val):

        # Converting data into Torch tensors
        x_train = torch.tensor(X_train, dtype=torch.long)
        y_train = torch.tensor(Y_train, dtype=torch.long)
        x_cv = torch.tensor(X_val, dtype=torch.long)
        y_cv = torch.tensor(Y_val, dtype=torch.long)
        
        # Converting dataset to a Torch Datset
        train = torch.utils.data.TensorDataset(x_train, y_train)
        valid = torch.utils.data.TensorDataset(x_cv, y_cv)
        
        # Initialising the DataLoaders
        train_dl = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) # shiffling enhance genralization avoids bias
        val_dl = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False) # No shuffling,epochs evaluations consistency
        return x_cv, y_cv, train_dl, val_dl

