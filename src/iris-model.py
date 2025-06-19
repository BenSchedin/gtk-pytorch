

# imports
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import plotly.express as px

# SCRIPT PARAMS
torch.manual_seed(123) # seed for reproducibility

train_proportion = 0.85 # percent of total samples to be used for training (1-train_proportion will be used for testing)

max_epochs = 1000
learning_rate = 1e-3


# loading data
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv('../data/raw/iris.data', header=None, names=columns)

print(data.shape)
print(data.head())

# encoding target
encoder = LabelEncoder()
data['species'] = encoder.fit_transform(data['species'])

print(data.shape)
print(data.head())

# creating train/test split
X = data.drop(columns=['species']).values
y = data['species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_proportion, random_state=123)

# converting resulting splits to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# nn implementation
class LinearModel(nn.Module):
    def __init__(self, n_features: int, h1: int, h2: int, n_targets: int):
        super().__init__()
        # initializing layers
        self.n_features = n_features # number of independent variables used for prediction
        self.h1 = h1 # number of nodes in the first hidden layer of the network
        self.h2 = h2 # number of nodes in the second hidden layer of the network
        self.n_targets = n_targets # number of dependent variable values that can be predicted

        # initializing layer connections
        self.fc1 = nn.Linear(n_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, n_targets)

    def forward(self, x):
        # defining activations for each forward pass between layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x

# instantiating model
model = LinearModel(n_features=4, h1=8, h2=8, n_targets=3)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# training loop
epochs = []
losses = []

for i in range(max_epochs):
    # passing data through the network
    y_pred = model.forward(X_train_tensor)

    # calculating loss
    loss = criterion(y_pred, y_train_tensor)
    losses.append(loss.item())
    epochs.append(i)

    # backprop
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# storing results in df and visualizing training performance
results = pd.DataFrame({
    'Training Epoch': pd.Series(epochs),
    'Loss': pd.Series(losses)
})

fig = px.line(
    results,
    x='Training Epoch',
    y='Loss'
)
fig.update_layout(
    title='Training Performance',
    xaxis_title='Training Epoch',
    yaxis_title='Cross Entropy Loss',
    template='plotly_dark'
)
fig.show()
