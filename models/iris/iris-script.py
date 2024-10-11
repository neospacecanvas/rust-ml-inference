# %%
import torch
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from alive_progress import alive_bar

# %%
class IrisDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        return x, y

# %%
class IrisClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(IrisClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# %%
# Load the Iris dataset
iris = load_iris()
features = iris.data
labels = iris.target

# Split the dataset into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
train_features = torch.tensor(train_features, dtype=torch.float32)
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_features = torch.tensor(test_features, dtype=torch.float32)
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Create the datasets
train_dataset = IrisDataset(train_features, train_labels)
test_dataset = IrisDataset(test_features, test_labels)

# Create the data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# %%
# Example usage
for features, labels in train_loader:
    print("Features:", features.shape)
    print("Labels:", labels.shape)
    break

# %%

input_size = 4
hidden_size = 10
num_classes = 3
model = IrisClassifier(input_size, hidden_size, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for features, labels in train_loader:
        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# Testing loop
model.eval()

# %%
def convert_to_onnx(model, input_size, file_name="iris_model.onnx"):
    # Import required libraries
    import torch
    import torch.onnx

    # Create a dummy input tensor
    dummy_input = torch.randn(1, input_size)

    # Export the model to ONNX format
    torch.onnx.export(model,               # model being run
                      dummy_input,         # model input (or a tuple for multiple inputs)
                      file_name,           # where to save the model
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=10,    # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['input'],   # the model's input names
                      output_names=['output'], # the model's output names
                      dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                    'output' : {0 : 'batch_size'}})
    
    print(f"Model has been converted to ONNX and saved as {file_name}")

# Example usage:
# convert_to_onnx(model, input_size=4)

# %%
convert_to_onnx(model, input_size=4)


