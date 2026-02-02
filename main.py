# Ensure torch_geometric is installed and available


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GraphSAGE
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def load_cicids2017(csv_path):
    df = pd.read_csv(csv_path)
    # Clean column names: strip whitespace and replace spaces with underscores
    df.columns = df.columns.str.strip().str.replace(' ', '_')
    # drop timestamps, select numeric features
    X = df.select_dtypes(include=np.number).replace([np.inf, -np.inf], np.nan).fillna(0).values

    # The column name 'Label' was identified in a previous step (cell 53753b9f) as the target column.
    # Ensure the column name used here matches the cleaned column names.
    y = df['label'].astype('category').cat.codes.values # Use 'Label' as indicated by previous output
    X = StandardScaler().fit_transform(X)
    return X, y

def construct_graph(X, k=5):
    from sklearn.neighbors import kneighbors_graph
    # Use k-NN to connect similar flows; adjust to session/IP graph logic
    A = kneighbors_graph(X, k, mode='connectivity', include_self=True)
    edge_index = torch.tensor(np.vstack(A.nonzero()), dtype=torch.long)
    x = torch.tensor(X, dtype=torch.float)
    return Data(x=x, edge_index=edge_index)

class GNNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.conv1 = GraphSAGE(in_channels=in_dim, hidden_channels=hidden_dim, num_layers=2)
        self.conv2 = GraphSAGE(in_channels=hidden_dim, hidden_channels=hidden_dim, num_layers=2)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        z = self.lin(x)
        return z

def sample_episode(X, y, num_way=5, k_shot=5, query_per=10):
    unique_classes = list(set(y))
    num_available_classes = len(unique_classes)
    # Ensure num_way does not exceed the actual number of unique classes
    if num_way > num_available_classes:
        num_way = num_available_classes

    classes = random.sample(unique_classes, num_way)
    support_idx, query_idx = [], []
    for c in classes:
        idx = np.where(y==c)[0]
        # Ensure enough samples are available for k_shot and query_per
        if len(idx) < k_shot + query_per:
             continue
        pick = np.random.choice(idx, k_shot + query_per, replace=False)
        support_idx.extend(pick[:k_shot])
        query_idx.extend(pick[k_shot:])
    return support_idx, query_idx, classes

def train_episode(model, optimizer, X, y):
    sup_idx, qry_idx, classes = sample_episode(X, y)
    if not classes or not qry_idx or not sup_idx:  # Skip episode if no classes were sampled or indices are empty
        return None, 0, 0 # Return None for loss and 0 for counts

    data = construct_graph(X)
    z = model(data)  # all node embeddings
    sup_z = z[sup_idx]; sup_y = y[sup_idx]
    qry_z = z[qry_idx]; qry_y = y[qry_idx]

    # compute prototypes
    prototypes = {}
    for c in classes:
        prototypes[c] = sup_z[sup_y==c].mean(dim=0)

    # classify queries
    logits = []
    correct = 0
    total = 0
    for qz, true in zip(qry_z, qry_y):
        total += 1
        dists = torch.stack([torch.sum((qz - prototypes[c])**2) for c in classes])
        pred_idx = torch.argmin(dists)
        pred = classes[pred_idx]
        if pred == true:
            correct += 1
        logits.append(-dists)

    logits = torch.stack(logits)  # shape (N_query, N_way)
    target = torch.tensor([classes.index(c) for c in qry_y])
    loss = F.cross_entropy(logits, target)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item(), correct, total

def evaluate_model(model, X, y, classes, k_shot=5, query_per=270):
    # Construct graph data
    data = construct_graph(X)

    # Obtain node embeddings
    z = model(data)

    # Sample support and query indices and labels
    sup_idx, qry_idx, sampled_classes = sample_episode(X, y, num_way=len(classes),
                                                       k_shot=k_shot, query_per=query_per)

    if not sampled_classes or not qry_idx or not sup_idx:
        # Ensure a consistent return type, even if there are no samples for evaluation
        return 0.0, 0.0, 0.0, 0.0, None, np.array([]), np.array([]), np.array([])

    # Extract support and query embeddings and labels
    sup_z = z[sup_idx]
    sup_y = y[sup_idx]
    qry_z = z[qry_idx]
    qry_y = y[qry_idx]

    # Compute prototypes
    prototypes = {}
    for c in sampled_classes:
        prototypes[c] = sup_z[sup_y == c].mean(dim=0)

    # Classify queries
    predicted_labels = []
    true_labels = []
    all_dists = [] # To store distances for ROC calculation

    for qz, true in zip(qry_z, qry_y):
        dists = torch.stack([torch.sum((qz - prototypes[c])**2) for c in sampled_classes])
        all_dists.append(-dists) # Store negative distances as scores (larger is better)
        pred_idx = torch.argmin(dists)
        pred = sampled_classes[pred_idx]
        predicted_labels.append(pred)
        true_labels.append(true)

    # Convert labels to numpy arrays for sklearn metrics
    true_labels_np = np.array(true_labels)
    predicted_labels_np = np.array(predicted_labels)

    # Handle cases where all_dists might be empty if no query samples were processed
    if not all_dists:
        predicted_scores_np = np.array([])
    else:
        # Ensure we have scores for each class for ROC curve; use softmax for probabilities
        predicted_scores_np = F.softmax(torch.stack(all_dists), dim=1).detach().cpu().numpy()

    # Calculate metrics
    accuracy = accuracy_score(true_labels_np, predicted_labels_np)
    # Use zero_division parameter to handle cases where there are no true or predicted positives
    precision = precision_score(true_labels_np, predicted_labels_np, average='weighted', zero_division=0)
    recall = recall_score(true_labels_np, predicted_labels_np, average='weighted', zero_division=0)
    f1 = f1_score(true_labels_np, predicted_labels_np, average='weighted', zero_division=0)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels_np, predicted_labels_np, labels=classes)

    return precision, recall, accuracy, f1, cm, true_labels_np, predicted_labels_np, predicted_scores_np

# Load data
X, y = load_cicids2017('D:\\Journal\\Meena\\Viji\\Unsw\\UNSW_NB15_testing-set.csv') # Ensure this path is correct

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_accuracies = []
val_accuracies = []
train_precisions = []
val_precisions = []
train_recalls = []
val_recalls = []
train_f1_scores = []
val_f1_scores = []

# For ROC and final confusion matrix
final_cm = None
final_val_true_labels = None
final_val_predicted_labels = None
final_val_predicted_scores = None # For ROC

epochs = 100
eval_interval = 10 # Changed from 1 to 10

model = GNNEncoder(in_dim=X_train.shape[1], hidden_dim=64, out_dim=32)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # Changed learning rate from 1e-3 to 1e-4

unique_classes = np.unique(y_train) # Use unique classes from the training set

for ep in range(1, epochs + 1):
    loss, correct, total = train_episode(model, optimizer, X_train, y_train)

    if loss is None: # Skip if sample_episode returned None due to insufficient samples
        continue

    if ep % eval_interval == 0:
        # Evaluate on training data
        train_p, train_r, train_acc, train_f1, _, _, _, _ = evaluate_model(model, X_train, y_train, classes=unique_classes)
        # Evaluate on validation data
        val_p, val_r, val_acc, val_f1, cm_val, true_val, pred_val, scores_val = evaluate_model(model, X_val, y_val, classes=unique_classes)

        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        train_precisions.append(train_p)
        val_precisions.append(val_p)
        train_recalls.append(train_r)
        val_recalls.append(val_r)
        train_f1_scores.append(train_f1)
        val_f1_scores.append(val_f1)

        # Store for final plots (from the last evaluation interval)
        final_cm = cm_val
        final_val_true_labels = true_val
        final_val_predicted_labels = pred_val
        final_val_predicted_scores = scores_val

        print(f'Episode {ep}, Loss: {loss:.4f}, Training Accuracy: {train_acc:.4f}, Validation Accuracy: {val_acc:.4f}')

# --- Plotting Results ---
episode_numbers = [ep * eval_interval for ep in range(1, len(train_accuracies) + 1)]

plt.figure(figsize=(15, 10))

# Plot 1: Training and Validation Accuracy
plt.subplot(2, 2, 1)
plt.plot(episode_numbers, train_accuracies, label='Training Accuracy')
plt.plot(episode_numbers, val_accuracies, label='Validation Accuracy')
plt.xlabel('Episode')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Episodes')
plt.legend()
plt.grid(True)

# Plot 2: Training and Validation Precision
plt.subplot(2, 2, 2)
plt.plot(episode_numbers, train_precisions, label='Training Precision')
plt.plot(episode_numbers, val_precisions, label='Validation Precision')
plt.xlabel('Episode')
plt.ylabel('Precision')
plt.title('Training and Validation Precision over Episodes')
plt.legend()
plt.grid(True)

# Plot 3: Training and Validation Recall
plt.subplot(2, 2, 3)
plt.plot(episode_numbers, train_recalls, label='Training Recall')
plt.plot(episode_numbers, val_recalls, label='Validation Recall')
plt.xlabel('Episode')
plt.ylabel('Recall')
plt.title('Training and Validation Recall over Episodes')
plt.legend()
plt.grid(True)

# Plot 4: Training and Validation F1 Score
plt.subplot(2, 2, 4)
plt.plot(episode_numbers, train_f1_scores, label='Training F1 Score')
plt.plot(episode_numbers, val_f1_scores, label='Validation F1 Score')
plt.xlabel('Episode')
plt.ylabel('F1 Score')
plt.title('Training and Validation F1 Score over Episodes')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot Confusion Matrix (from the last evaluation)
if final_cm is not None and len(unique_classes) > 0:
    plt.figure(figsize=(8, 6))
    sns.heatmap(final_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=unique_classes, yticklabels=unique_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Validation Data - Last Episode)')
    plt.show()
else:
    print("Could not generate Confusion Matrix. No sufficient data.")

# Plot ROC Curve (if applicable for multi-class, otherwise for binary) for validation data
if final_val_true_labels is not None and final_val_predicted_scores.size > 0 and len(unique_classes) > 1:
    plt.figure(figsize=(8, 6))
    # For multi-class, plot one-vs-rest ROC curves
    for i, class_id in enumerate(unique_classes):
        if i < final_val_predicted_scores.shape[1]: # Ensure class_id corresponds to a column in scores
            fpr, tpr, _ = roc_curve(final_val_true_labels == class_id, final_val_predicted_scores[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'ROC curve of class {class_id} (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Validation Data - Last Episode)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()
elif final_val_true_labels is not None and final_val_predicted_scores.size > 0 and len(unique_classes) == 1:
    # Special handling for single class, ROC is not meaningful in traditional sense
    print("ROC curve cannot be plotted for a single class.")
else:
    print("Could not generate ROC Curve. No sufficient data.")