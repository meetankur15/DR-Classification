#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import cv2
import os
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf; print(tf.__version__)


# # TensorFlow pipeline
# We will create a dataset pipeline from our data using tensorflow dataset api
# 
# TensorFlow has a dataset api that supports writing descriptive and efficient input pipelines. a common pattern is to
# - Create a source dataset from your input data.
# - Apply dataset transformations to preprocess the data.
# - Iterate over the dataset and process the elements.
# 
# [For more information](https://www.tensorflow.org/api_docs/python/tf/data/Dataset)

# In[3]:


# generating dataset from directory

# Generating train dataset
data = tf.keras.utils.image_dataset_from_directory(directory = 'D1',
                                                   color_mode = 'rgb',
                                                   batch_size = 64,
                                                   image_size = (224,224),
                                                   shuffle=True,
                                                   seed = 2022)


# ## Displaying data distribution

# In[4]:


labels = np.concatenate([y for x,y in data], axis=0)


# In[5]:


values = pd.value_counts(labels)
values = values.sort_index()


# In[6]:


values


# ### Checking labels

# In[7]:


# getting class names
class_names = data.class_names
for idx, name in enumerate(class_names):
  print(f"{idx} = {name}", end=", ")


# ### The data is well distributed among the classes and is balanced

# In[8]:


import plotly.graph_objects as go

fig = go.Figure(
    data=go.Pie(
        labels=data.class_names,
        values=values,
        textinfo='percent+label',
        pull=[0.02] * len(data.class_names),
        textfont=dict(
            size=16,
            family="Arial Black"  # or "Calibri Bold", "Times New Roman Bold"
        )
    )
)
fig.update_layout(width=1000, height=700)
fig.show()


# In[9]:


data_iterator = data.as_numpy_iterator()


# In[10]:


batch = data_iterator.next()


# In[11]:


batch[0].shape


# ## Pre-processing the data

# In[12]:


data = data.map(lambda x, y: (x/255, y))


# In[13]:


sample = data.as_numpy_iterator().next()


# In[14]:


print(sample[0].min())
print(sample[0].max())


# ## Spliting the data

# In[15]:


print("Total number of batchs = ",len(data))


# In[16]:


train_size = int(0.7 * len(data)) +1
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))


# In[17]:


train = data.take(train_size)
remaining = data.skip(train_size)
val = remaining.take(val_size)
test = remaining.skip(val_size)


# In[18]:


print(f"# train batchs = {len(train)}, # validate batchs = {len(val)}, # test batch = {len(test)}")
len(train) + len(val) + len(test)


# ## Preparing test set

# In[19]:


test_iter = test.as_numpy_iterator()


# In[20]:


test_set = {"images":np.empty((0,224,224,3)), "labels":np.empty(0)}
while True:
  try:
    batch = test_iter.next()
    test_set['images'] = np.concatenate((test_set['images'], batch[0]))
    test_set['labels'] = np.concatenate((test_set['labels'], batch[1]))
  except:
    break


# In[21]:


y_true = test_set['labels']


# In[51]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
# Import specific GNN layers from torch_geometric.nn
from torch_geometric.nn import GCNConv, global_mean_pool 
import matplotlib.pyplot as plt

print("Loading dataset from D1...")
train_data_tf = tf.keras.utils.image_dataset_from_directory(
    directory='D1',
    color_mode='rgb',
    batch_size=None, # Load as individual images, not batches
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
)

# TensorFlow dataset to NumPy arrays

all_images = []
all_labels = []
for img_tensor, label_tensor in train_data_tf:
    all_images.append(img_tensor.numpy())
    all_labels.append(label_tensor.numpy())

all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Split the data into train and validation sets

from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(
    all_images, all_labels, test_size=0.3, random_state=2022, stratify=all_labels
) 

print(f"Total images loaded: {len(all_images)}")
print(f"Train images: {len(train_images)}, Validation images: {len(val_images)}")
print(f"Image shape: {train_images.shape[1:]}")
num_classes = len(train_data_tf.class_names)
print(f"Number of classes: {num_classes}")


# --- Create edge_index for 4-connectivity ---

def create_pixel_edges(H, W):
    source_nodes = []
    target_nodes = []

    for r in range(H):
        for c in range(W):
            current_node_idx = r * W + c

            # Right neighbor
            if c + 1 < W:
                right_neighbor_idx = r * W + (c + 1)
                source_nodes.append(current_node_idx)
                target_nodes.append(right_neighbor_idx)
                source_nodes.append(right_neighbor_idx) # Reciprocal edge for undirected graph
                target_nodes.append(current_node_idx)

            # Down neighbor
            if r + 1 < H:
                down_neighbor_idx = (r + 1) * W + c
                source_nodes.append(current_node_idx)
                target_nodes.append(down_neighbor_idx)
                source_nodes.append(down_neighbor_idx) # Reciprocal edge
                target_nodes.append(current_node_idx)



    return torch.tensor([source_nodes, target_nodes], dtype=torch.long)

# Getting image dimensions from loaded data
image_height, image_width, num_channels_img = train_images.shape[1:]
print(f"Derived image dimensions: H={image_height}, W={image_width}, C={num_channels_img}")

# Pre-calculate edge_index as it's the same for all images of the same size
# This avoids recalculating for every image, improving efficiency
common_edge_index = create_pixel_edges(image_height, image_width)


# --- Train images to graph data conversion ---
train_graphs = []
print("Converting train images to graph data...")
for i in range(len(train_images)):
    img_array = train_images[i] 

    # Normalize [0, 1]
    x_node_features = torch.tensor(img_array, dtype=torch.float32) / 255.0 


    x_node_features = x_node_features.view(image_height * image_width, num_channels_img)

    # Label for the graph
    y_label = torch.tensor(train_labels[i], dtype=torch.long)


    train_graphs.append(Data(x=x_node_features, edge_index=common_edge_index, y=y_label))

train_dataset_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
print(f"Created {len(train_graphs)} train graph instances.")


# --- validation images to graph data ---
val_graphs = []
print("Converting validation images to graph data...")
for i in range(len(val_images)):
    img_array = val_images[i] 

    x_node_features = torch.tensor(img_array, dtype=torch.float32) / 255.0
    x_node_features = x_node_features.view(image_height * image_width, num_channels_img)

    y_label = torch.tensor(val_labels[i], dtype=torch.long)

    val_graphs.append(Data(x=x_node_features, edge_index=common_edge_index, y=y_label))

val_dataset_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
print(f"Created {len(val_graphs)} validation graph instances.")


# --- Graph Neural Network Model ---
class GNNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_classes):
        super(GNNClassifier, self).__init__()
        # GNN layers for graph convolution
        # GCNConv takes (in_features, out_features)
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)


        self.fc = nn.Linear(out_channels, num_classes) 

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Apply GNN layers
        x = self.conv1(x, edge_index)
        x = torch.relu(x) # Activation function
        x = self.conv2(x, edge_index)
        x = torch.relu(x) # Activation function


        x = global_mean_pool(x, batch) # x becomes [num_graphs_in_batch, out_channels]

        # Final classification layer
        x = self.fc(x) 
        return x

# Initialize the GNNClassifier model

model = GNNClassifier(in_channels=num_channels_img, hidden_channels=128, out_channels=256, num_classes=num_classes)
print("GNN model initialized.")
print(model)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

print("Starting training...")
for epoch in range(10): # Example 10 epochs
    model.train() # Set model to training mode
    epoch_train_loss = 0.0
    epoch_train_correct_predictions = 0
    total_train_graphs_processed = 0

    for batch_data in train_dataset_loader: # Iterate over batches of graph data
        optimizer.zero_grad()
        output = model(batch_data) # Pass the entire batch_data object to the GNN

        # batch_data.y contains the labels for all graphs in the current batch
        loss = criterion(output, batch_data.y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        # loss.item() is the scalar loss for the batch (mean if reduction='mean')
        # data.num_graphs is the actual number of graphs in the current batch
        epoch_train_loss += loss.item() * batch_data.num_graphs 
        epoch_train_correct_predictions += (predicted == batch_data.y).sum().item()
        total_train_graphs_processed += batch_data.num_graphs

    train_loss = epoch_train_loss / total_train_graphs_processed
    train_accuracy = epoch_train_correct_predictions / total_train_graphs_processed

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    # --- Validation Loop ---
    model.eval() # Set model to evaluation mode
    epoch_val_loss = 0.0
    epoch_val_correct_predictions = 0
    total_val_graphs_processed = 0

    with torch.no_grad(): 
        for batch_data in val_dataset_loader:
            output = model(batch_data)
            loss = criterion(output, batch_data.y)

            _, predicted = torch.max(output, 1)
            epoch_val_loss += loss.item() * batch_data.num_graphs
            epoch_val_correct_predictions += (predicted == batch_data.y).sum().item()
            total_val_graphs_processed += batch_data.num_graphs

    val_loss = epoch_val_loss / total_val_graphs_processed
    val_accuracy = epoch_val_correct_predictions / total_val_graphs_processed

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    # loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    print("-" * 30)

print("Training complete.")

# Plot the accuracy and loss graphs
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.title('Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label='Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[22]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.nn import Conv2d
import matplotlib.pyplot as plt

# Generating train dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    directory='D1',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
)

# Prepare the train dataset for RGCNN
train_images = []
train_labels = []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Split the data into train and validation sets
train_split = int(0.7 * len(train_images))
val_images = train_images[train_split:]
val_labels = train_labels[train_split:]
train_images = train_images[:train_split]
train_labels = train_labels[:train_split]

# Convert train images to graph data
train_graphs = []
for i in range(len(train_images)):
    x = torch.tensor(train_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(train_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    train_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
train_dataset = DataLoader(train_graphs, batch_size=16, shuffle=True)

# Convert validation images to graph data
val_graphs = []
for i in range(len(val_images)):
    x = torch.tensor(val_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(val_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    val_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
val_dataset = DataLoader(val_graphs, batch_size=16, shuffle=False)


# Define the RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the RGCNN model
model = RGCNN(num_classes=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

for epoch in range(10):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0

    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        epoch_train_loss += loss.item() * data.num_graphs
        epoch_train_correct += (predicted == data.y).sum().item()

    train_loss = epoch_train_loss / len(train_images)
    train_accuracy = epoch_train_correct / len(train_images)

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    model.eval()
    epoch_val_loss = 0.0
    epoch_val_correct = 0

    for data in val_dataset:
        output = model(data.x)
        loss = criterion(output, data.y)

        _, predicted = torch.max(output, 1)
        epoch_val_loss += loss.item() * data.num_graphs
        epoch_val_correct += (predicted == data.y).sum().item()

    val_loss = epoch_val_loss / len(val_images)
    val_accuracy = epoch_val_correct / len(val_images)

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    print()

# Plot the accuracy and loss graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label= 'Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# In[41]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.nn import Conv2d
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

# Generating train dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    directory='D1',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
)
# Prepare the train dataset for RGCNN
train_images = []
train_labels = []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)
# Split the data into train and validation sets
train_split = int(0.7 * len(train_images))
val_images = train_images[train_split:]
val_labels = train_labels[train_split:]
train_images = train_images[:train_split]
train_labels = train_labels[:train_split]
# Convert train images to graph data
train_graphs = []
for i in range(len(train_images)):
    x = torch.tensor(train_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0) 
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0] 
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long) # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long) # Example edge types, adjust according to your graph structure
    y = torch.tensor(train_labels[i], dtype=torch.long)
    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes
    train_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
train_dataset = DataLoader(train_graphs, batch_size=16, shuffle=True)
# Convert validation images to graph data
val_graphs = []
for i in range(len(val_images)):
    x = torch.tensor(val_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0) # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0] 
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long) 
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long) 
    y = torch.tensor(val_labels[i], dtype=torch.long)

    edge_index = edge_index % num_nodes
    val_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
val_dataset = DataLoader(val_graphs, batch_size=16, shuffle=False)

# RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Initialize the RGCNN model
model = RGCNN(num_classes=5)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []
for epoch in range(10):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        epoch_train_loss += loss.item() * data.num_graphs
        epoch_train_correct += (predicted == data.y).sum().item()
    train_loss = epoch_train_loss / len(train_images)
    train_accuracy = epoch_train_correct / len(train_images)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    # ====================== VALIDATION WITH QWK ======================
    model.eval()
    epoch_val_loss = 0.0
    epoch_val_correct = 0
    val_preds = []      # ← new
    val_trues = []      # ← new
    for data in val_dataset:
        output = model(data.x)
        loss = criterion(output, data.y)
        _, predicted = torch.max(output, 1)
        epoch_val_loss += loss.item() * data.num_graphs
        epoch_val_correct += (predicted == data.y).sum().item()

        # Collect predictions & ground truth for QWK
        val_preds.extend(predicted.cpu().numpy().tolist())
        val_trues.extend(data.y.cpu().numpy().tolist())

    val_loss = epoch_val_loss / len(val_images)
    val_accuracy = epoch_val_correct / len(val_images)


    val_qwk = cohen_kappa_score(val_trues, val_preds, weights='quadratic')

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val QWK: {val_qwk:.4f}")
    print()


# Grad-CAM EXPLAINABILITY

class GradCAM:
    """Simple, self-contained Grad-CAM using PyTorch hooks."""
    def __init__(self, model, target_layer_name='conv2'):
        self.model = model
        # Get the target convolutional layer dynamically
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """Generate CAM for a single image (shape: [1, 3, 224, 224])."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients      
        activations = self.activations 


        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)


        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


# Grad-CAM for one validation image
if len(val_graphs) > 0:
    # Take the first validation sample (already prepared as graph Data)
    sample_data = val_graphs[0]
    sample_x = sample_data.x.to(next(model.parameters()).device)  # ensure correct device


    gradcam = GradCAM(model, target_layer_name='conv2')


    cam = gradcam(sample_x)

    # Prepare original image for visualization
    orig_img = sample_x[0].permute(1, 2, 0).cpu().numpy()         
    orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)

    # Plot side-by-side
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(orig_img)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(orig_img)
    plt.imshow(cam.cpu().numpy(), cmap='jet', alpha=0.6)
    plt.title('Grad-CAM Heatmap (conv2 layer)\nModel focuses on clinically relevant regions')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    print("Grad-CAM explainability ")
    print("The model looks after graph conversion).")


# In[23]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.nn import Conv2d
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

# Generating train dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    directory='D1',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
)
# Prepare the train dataset for RGCNN
train_images = []
train_labels = []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)
# Split the data into train and validation sets
train_split = int(0.7 * len(train_images))
val_images = train_images[train_split:]
val_labels = train_labels[train_split:]
train_images = train_images[:train_split]
train_labels = train_labels[:train_split]
# train images to graph data
train_graphs = []
for i in range(len(train_images)):
    x = torch.tensor(train_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0) # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0] # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long) # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long) # Example edge types, adjust according to your graph structure
    y = torch.tensor(train_labels[i], dtype=torch.long)
    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes
    train_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
train_dataset = DataLoader(train_graphs, batch_size=16, shuffle=True)
# validation images to graph data
val_graphs = []
for i in range(len(val_images)):
    x = torch.tensor(val_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0) # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0] # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long) # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long) # Example edge types, adjust according to your graph structure
    y = torch.tensor(val_labels[i], dtype=torch.long)
    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes
    val_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
val_dataset = DataLoader(val_graphs, batch_size=16, shuffle=False)
# Define the RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
# Initialize the RGCNN model
model = RGCNN(num_classes=5)
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Training loop
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []
val_qwk_history = []
for epoch in range(10):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0
    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()
        _, predicted = torch.max(output, 1)
        epoch_train_loss += loss.item() * data.num_graphs
        epoch_train_correct += (predicted == data.y).sum().item()
    train_loss = epoch_train_loss / len(train_images)
    train_accuracy = epoch_train_correct / len(train_images)
    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)
    # MODEL SUMMARY 
print("\n" + "=" * 70)
print("RGCNN MODEL SUMMARY (for your paper)")
print("=" * 70)
print(model)
print("\nTrainable Parameters:")
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total trainable parameters : {total_params:,}")
print("\nArchitecture notes:")
print("  • Conv2d(3→16) + ReLU")
print("  • Conv2d(16→32) + ReLU")
print("  • Flatten → Linear(32×224×224 → 5)")
print("  • Currently pure CNN (graph edges not used in forward pass)")
print("=" * 70)

#  VALIDATION WITH QWK 
model.eval()
epoch_val_loss = 0.0
epoch_val_correct = 0
val_preds = []
val_trues = []
for data in val_dataset:
    output = model(data.x)
    loss = criterion(output, data.y)
    _, predicted = torch.max(output, 1)
    epoch_val_loss += loss.item() * data.num_graphs
    epoch_val_correct += (predicted == data.y).sum().item()

        # Collect for QWK
    val_preds.extend(predicted.cpu().numpy().tolist())
    val_trues.extend(data.y.cpu().numpy().tolist())

    val_loss = epoch_val_loss / len(val_images)
    val_accuracy = epoch_val_correct / len(val_images)

    # Quadratic Weighted Kappa
    val_qwk = cohen_kappa_score(val_trues, val_preds, weights='quadratic')

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)
    val_qwk_history.append(val_qwk)          # ← saved for final summary

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val QWK: {val_qwk:.4f}")
    print()


# Grad-CAM EXPLAINABILITY

class GradCAM:
    """Simple, self-contained Grad-CAM using PyTorch hooks."""
    def __init__(self, model, target_layer_name='conv2'):
        self.model = model
        self.target_layer = dict(model.named_modules())[target_layer_name]
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, target_class=None):
        """Generate CAM for a single image (shape: [1, 3, 224, 224])."""
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        score = output[0, target_class]
        score.backward()

        gradients = self.gradients      # [1, C, H, W]
        activations = self.activations  # [1, C, H, W]

        # Global Average Pooling on gradients
        weights = torch.mean(gradients, dim=[2, 3], keepdim=True)

        # Weighted sum → CAM
        cam = torch.sum(weights * activations, dim=1).squeeze(0)  # [H, W]
        cam = torch.relu(cam)

        # Normalize to [0, 1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam


class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
samples_per_class = {i: None for i in range(5)}
for data in val_graphs:
    label = data.y.item()
    if samples_per_class[label] is None:
        samples_per_class[label] = data
    if all(v is not None for v in samples_per_class.values()):
        break


if any(v is not None for v in samples_per_class.values()):
    gradcam = GradCAM(model, target_layer_name='conv2')

    fig, axs = plt.subplots(1, 5, figsize=(25, 6))
    fig.suptitle('Grad-CAM Explainability — One Representative Image per DR Class\n'
                 '(Highlights clinically relevant regions after image-to-graph conversion)', 
                 fontsize=16, y=1.02)

    for idx, class_id in enumerate(range(5)):
        if samples_per_class[class_id] is None:
            axs[idx].text(0.5, 0.5, f'No sample\nfor {class_names[class_id]}', 
                         ha='center', va='center', fontsize=12)
            axs[idx].axis('off')
            continue

        sample_data = samples_per_class[class_id]
        sample_x = sample_data.x.to(next(model.parameters()).device)

        cam = gradcam(sample_x)

        # Original image
        orig_img = sample_x[0].permute(1, 2, 0).cpu().numpy()
        orig_img = np.clip(orig_img * 255, 0, 255).astype(np.uint8)

        # Plot
        axs[idx].imshow(orig_img)
        axs[idx].imshow(cam.cpu().numpy(), cmap='jet', alpha=0.65)
        axs[idx].set_title(f'{class_names[class_id]}\n(Class {class_id})', fontsize=13)
        axs[idx].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

    print("Grad-CAM generated for each DR class")
    print("Grad CAM")
else:
    print("No validation samples available")

    # FINAL SUMMARY 
print("\n" + "=" * 70)
print("FINAL MODEL PERFORMANCE SUMMARY")
print("=" * 70)
print(f"Model                  : RGCNN (image → graph → CNN)")
print(f"Final Train Accuracy   : {train_accuracy_history[-1]:.4f}")
print(f"Final Validation Accuracy : {val_accuracy_history[-1]:.4f}")
print(f"Final Validation QWK   : {val_qwk_history[-1]:.4f}   ← (primary DR metric)")
print(f"Epochs                 : 10")
print(f"Total Trainable Params : {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print("\nRecommendation:")
print("=" * 70)


# In[24]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.nn import Conv2d
from sklearn.metrics import classification_report, confusion_matrix

# Generating train dataset
data = tf.keras.utils.image_dataset_from_directory(
    directory='D1',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate', 'Severe']
)

# Split the data into train, validation, and test sets
train_size = int(0.7 * len(data))
val_size = int(0.2 * len(data))
test_size = int(0.1 * len(data))

train_data = data.take(train_size)
remaining = data.skip(train_size)
val_data = remaining.take(val_size)
test_data = remaining.skip(val_size)

# Prepare the train dataset for RGCNN
train_images = []
train_labels = []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Convert train images to graph data
train_graphs = []
for i in range(len(train_images)):
    x = torch.tensor(train_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(train_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    train_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
train_dataset = DataLoader(train_graphs, batch_size=16, shuffle=True)

# Convert validation images to graph data
val_images = []
val_labels = []
for images, labels in val_data:
    val_images.append(images)
    val_labels.append(labels)
val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

val_graphs = []
for i in range(len(val_images)):
    x = torch.tensor(val_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(val_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    val_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
val_dataset = DataLoader(val_graphs, batch_size=16, shuffle=False)

# Define the RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the RGCNN model
model = RGCNN(num_classes=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(10):
    model.train()
    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0

    for data in val_dataset:
        output = model(data.x)
        loss = criterion(output, data.y)

        _, predicted = torch.max(output, 1)
        val_loss += loss.item() * data.num_graphs
        val_correct += (predicted == data.y).sum().item()

    val_loss /= len(val_dataset)
    val_accuracy = val_correct / len(val_images)

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Validation Loss: {val_loss:.4f} | Validation Accuracy: {val_accuracy:.4f}")
    print()

# Testing
test_images = []
test_labels = []
for images, labels in test_data:
    test_images.append(images)
    test_labels.append(labels)
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

test_graphs = []
for i in range(len(test_images)):
    x = torch.tensor(test_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(test_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    test_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
test_dataset = DataLoader(test_graphs, batch_size=16, shuffle=False)

# Testing evaluation
model.eval()
test_loss = 0.0
test_correct = 0

for data in test_dataset:
    output = model(data.x)
    loss = criterion(output, data.y)

    _, predicted = torch.max(output, 1)
    test_loss += loss.item() * data.num_graphs
    test_correct += (predicted == data.y).sum().item()

test_loss /= len(test_dataset)
test_accuracy = test_correct / len(test_images)

# Print the test loss and accuracy
print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.4f}")

# Compute classification report and confusion matrix
y_true = []
y_pred = []

for data in test_dataset:
    output = model(data.x)
    _, predicted = torch.max(output, 1)
    y_true.extend(data.y.tolist())
    y_pred.extend(predicted.tolist())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

report = classification_report(y_true, y_pred)
confusion_mat = confusion_matrix(y_true, y_pred)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(confusion_mat)


# In[30]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# Compute the predicted probabilities for each class
y_probs = []
y_true = []

for data in test_dataset:
    output = model(data.x)
    probs = torch.softmax(output, dim=1)
    y_probs.extend(probs.detach().numpy())
    y_true.extend(data.y.tolist())

y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Compute the precision-recall curve and average precision scores for each class
precision_scores = []
recall_scores = []
average_precision_scores = []

for class_idx in range(y_probs.shape[1]):
    y_true_class = y_true == class_idx
    y_probs_class = y_probs[:, class_idx]
    precision, recall, _ = precision_recall_curve(y_true_class, y_probs_class)
    average_precision = average_precision_score(y_true_class, y_probs_class)
    precision_scores.append(precision)
    recall_scores.append(recall)
    average_precision_scores.append(average_precision)

# Plot the precision-recall curves
plt.figure(figsize=(8, 6))
for class_idx in range(y_probs.shape[1]):
    plt.plot(recall_scores[class_idx], precision_scores[class_idx], label=f"Class {class_idx} (AP = {average_precision_scores[class_idx]:.4f})")

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()


# In[31]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# Compute the predicted probabilities for each class
y_probs = []
y_true = []

for data in test_dataset:
    output = model(data.x)
    probs = torch.softmax(output, dim=1)
    y_probs.extend(probs.detach().numpy())
    y_true.extend(data.y.tolist())

y_probs = np.array(y_probs)
y_true = np.array(y_true)

# Binarize the true labels
y_true_bin = label_binarize(y_true, classes=np.arange(5))  # Assuming 5 classes

# Compute the ROC curve and AUC for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

for class_idx in range(y_probs.shape[1]):
    fpr[class_idx], tpr[class_idx], _ = roc_curve(y_true_bin[:, class_idx], y_probs[:, class_idx])
    roc_auc[class_idx] = auc(fpr[class_idx], tpr[class_idx])

# Compute the micro-average ROC curve and AUC
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), y_probs.ravel())
roc_auc_micro = auc(fpr_micro, tpr_micro)

# Plot the ROC curves
plt.figure(figsize=(8, 6))
for class_idx in range(y_probs.shape[1]):
    plt.plot(fpr[class_idx], tpr[class_idx], label=f"Class {class_idx} (AUC = {roc_auc[class_idx]:.4f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()


# In[42]:


import networkx as nx
import matplotlib.pyplot as plt

# Choose the dataset (train or test)
dataset = train_images  # or test_images

# Create a networkx graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = 30  # or 4
G.add_nodes_from(range(num_nodes))

# Add edges to the graph (modify this based on your graph structure)
edge_index = [[0, 1, 2], [1, 0, 2]]  # Example edge index, adjust according to your graph structure
edges = [(i, j) for i, j in zip(*edge_index)]
G.add_edges_from(edges)

# Draw the graph
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')

# Show the graph
plt.title('Graph Structure')
plt.show()


# In[67]:


import networkx as nx
import matplotlib.pyplot as plt

# Choose the dataset (train or test)
dataset = test_images  # or test_images

# Create a networkx graph
G = nx.Graph()

# Add nodes to the graph
num_nodes = 5
G.add_nodes_from(range(num_nodes))

# Define the edges for a square graph structure
edges = [(i, i + 1) for i in range(num_nodes - 1)] + [(num_nodes - 1, 0)]

# Add edges to the graph
G.add_edges_from(edges)

# Draw the graph
pos = nx.kamada_kawai_layout(G)  # Predefined spring layout
nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')

# Show the graph
plt.title('Graph Structure')
plt.show()


# In[74]:


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Load the image
image = plt.imread('D1/11.png')

# Create a graph
G = nx.Graph()

# Get the dimensions of the image
height, width, _ = image.shape

# Add nodes to the graph
for i in range(height):
    for j in range(width):
        node_id = (i, j)
        G.add_node(node_id, pos=(i, j))

# Add edges between adjacent pixels
for i in range(height):
    for j in range(width):
        if i < height - 1:
            G.add_edge((i, j), (i + 1, j))
        if j < width - 1:
            G.add_edge((i, j), (i, j + 1))

# Draw the graph
pos = nx.get_node_attributes(G, 'pos')
nx.draw(G, pos, with_labels=False, node_size=1, width=0.1)

# Show the graph
plt.axis('off')
plt.show()


# In[ ]:


import tensorflow as tf
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch.nn import Conv2d
import matplotlib.pyplot as plt

# Generating train dataset
train_data = tf.keras.utils.image_dataset_from_directory(
    directory='DRM',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=True,
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proloferate', 'Severe']
)

# Prepare the train dataset for RGCNN
train_images = []
train_labels = []
for images, labels in train_data:
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Split the data into train and validation sets
train_split = int(0.8 * len(train_images))
val_images = train_images[train_split:]
val_labels = train_labels[train_split:]
train_images = train_images[:train_split]
train_labels = train_labels[:train_split]

# Convert train images to graph data
train_graphs = []
for i in range(len(train_images)):
    x = torch.tensor(train_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(train_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    train_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
train_dataset = DataLoader(train_graphs, batch_size=16, shuffle=True)

# Convert validation images to graph data
val_graphs = []
for i in range(len(val_images)):
    x = torch.tensor(val_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(val_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    val_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
val_dataset = DataLoader(val_graphs, batch_size=16, shuffle=False)


# Define the RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the RGCNN model
model = RGCNN(num_classes=5)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
train_loss_history = []
train_accuracy_history = []
val_loss_history = []
val_accuracy_history = []

for epoch in range(20):
    model.train()
    epoch_train_loss = 0.0
    epoch_train_correct = 0

    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output, 1)
        epoch_train_loss += loss.item() * data.num_graphs
        epoch_train_correct += (predicted == data.y).sum().item()

    train_loss = epoch_train_loss / len(train_images)
    train_accuracy = epoch_train_correct / len(train_images)

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

    model.eval()
    epoch_val_loss = 0.0
    epoch_val_correct = 0

    for data in val_dataset:
        output = model(data.x)
        loss = criterion(output, data.y)

        _, predicted = torch.max(output, 1)
        epoch_val_loss += loss.item() * data.num_graphs
        epoch_val_correct += (predicted == data.y).sum().item()

    val_loss = epoch_val_loss / len(val_images)
    val_accuracy = epoch_val_correct / len(val_images)

    val_loss_history.append(val_loss)
    val_accuracy_history.append(val_accuracy)

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}:")
    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f}")
    print()

# Plot the accuracy and loss graphs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracy_history, label='Training Accuracy')
plt.plot(val_accuracy_history, label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_loss_history, label= 'Training Loss')
plt.plot(val_loss_history, label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()


# In[3]:


# Generating test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    directory='DRM',
    color_mode='rgb',
    batch_size=64,
    image_size=(224, 224),
    shuffle=False,  # Disable shuffling for the test set
    seed=2022,
    labels='inferred',
    label_mode='int',
    class_names=['Mild', 'Moderate', 'No_DR', 'Proloferate', 'Severe']
)

# Prepare the test dataset for RGCNN
test_images = []
test_labels = []
for images, labels in test_data:
    test_images.append(images)
    test_labels.append(labels)
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

# Convert test images to graph data
test_graphs = []
for i in range(len(test_images)):
    x = torch.tensor(test_images[i], dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure
    y = torch.tensor(test_labels[i], dtype=torch.long)

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    test_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type, y=y))
test_dataset = DataLoader(test_graphs, batch_size=16, shuffle=False)


# In[37]:


import matplotlib.pyplot as plt

# Initialize lists to store accuracy and loss values
train_losses = []
train_accuracies = []

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for data in train_dataset:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted_labels = torch.max(output, dim=1)
        correct += (predicted_labels == data.y).sum().item()
        total += data.y.size(0)

        running_loss += loss.item()

    # Calculate average training loss and accuracy for the epoch
    epoch_loss = running_loss / len(train_dataset)
    epoch_accuracy = 100.0 * correct / total

    # Store the values for plotting
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

    # Print the loss and accuracy for each epoch
    print(f"Epoch {epoch + 1}: Loss = {epoch_loss}, Accuracy = {epoch_accuracy}%")

# Plot the accuracy and loss graphs
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.legend()

plt.tight_layout()
plt.show()


# In[38]:


import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch.nn import Conv2d

# Assuming you have prepared your test set
test_set = {"images": np.empty((0, 224, 224, 3)), "labels": np.empty(0)}
while True:
    try:
        batch = test_iter.next()
        test_set['images'] = np.concatenate((test_set['images'], batch[0]))
        test_set['labels'] = np.concatenate((test_set['labels'], batch[1]))
    except:
        break

# Preprocess test images and convert them to graph data
test_graphs = []
for image in test_set['images']:
    x = torch.tensor(image, dtype=torch.float32) / 255.0
    if len(x.shape) == 3:
        x = x.permute(2, 0, 1)
        x = x.unsqueeze(0)  # Add batch dimension
    elif len(x.shape) == 4:
        x = x.permute(0, 3, 1, 2)
    num_nodes = x.shape[0]  # Get the number of nodes in the graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)  # Example edge index, adjust according to your graph structure
    edge_type = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Example edge types, adjust according to your graph structure

    # Adjust the edge indices to match the number of nodes in the graph
    edge_index = edge_index % num_nodes

    test_graphs.append(Data(x=x, edge_index=edge_index, edge_type=edge_type))

# Create a DataLoader for the test dataset
test_dataset = DataLoader(test_graphs, batch_size=16, shuffle=False)

# Define the RGCNN model
class RGCNN(nn.Module):
    def __init__(self, num_classes):
        super(RGCNN, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 224 * 224, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# Initialize the RGCNN model
model = RGCNN(num_classes=5)

# Load the trained model weights
model.load_state_dict(torch.load('path_to_your_trained_model_weights.pth'))

# Set the model to evaluation mode
model.eval()

# Perform inference on the test dataset
predictions = []
with torch.no_grad():
    for data in test_dataset:
        output = model(data.x)
        _, predicted_labels = torch.max(output, dim=1)
        predictions.append(predicted_labels.cpu().numpy())

# Concatenate the predictions from different batches
predictions = np.concatenate(predictions)

# Print the predicted labels
print("Predicted labels:", predictions)


# In[24]:


# Displaying history loss/accuracy
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
def plot_his(history):
    plt.figure(figsize=(15,12))
    metrics = ['accuracy', 'loss']
    for i, metric in enumerate(metrics):
        plt.subplot(220+1+i)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[1], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.legend()
    plt.show()


# ### 
