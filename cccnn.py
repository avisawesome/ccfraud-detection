import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc, roc_curve
from imblearn.over_sampling import SMOTE

# For PyTorch implementation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# For TensorFlow implementation
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

np.random.seed(42)
torch.manual_seed(42)
tf.random.set_seed(42)

def load_data():
    #from Kaggle dataset "Credit Card Fraud Detection"
    df = pd.read_csv('creditcard.csv')
    
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud cases: {df['Class'].sum()}")
    print(f"Fraud percentage: {df['Class'].mean() * 100:.4f}%")
    
    return df

def preprocess_data(df):
    X = df.drop('Class', axis=1).values
    y = df['Class'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    X_train_resampled = X_train_resampled.reshape(X_train_resampled.shape[0], X_train_resampled.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    
    print(f"Training data shape after SMOTE: {X_train_resampled.shape}")
    print(f"Validation data shape: {X_val.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    return X_train_resampled, y_train_resampled, X_val, y_val, X_test, y_test, scaler

# Custom PyTorch Dataset
class FraudDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# PyTorch CNN Model
class FraudCNN(nn.Module):
    def __init__(self, input_dim):
        super(FraudCNN, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        conv_output_size = input_dim // 4 * 64
        
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def train_pytorch_model(X_train, y_train, X_val, y_val, input_dim):
    train_dataset = FraudDataset(X_train, y_train)
    val_dataset = FraudDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    model = FraudCNN(input_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 25
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device).view(-1, 1)
            
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device).view(-1, 1)
                
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_true.extend(labels.cpu().numpy().flatten())
        
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    
    model.load_state_dict(best_model_state)
    
    return model

# Function to create and train TensorFlow model
def train_tensorflow_model(X_train, y_train, X_val, y_val):
    model = Sequential([
        Conv1D(32, kernel_size=3, padding='same', activation='relu', input_shape=(X_train.shape[1], 1)),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint('best_fraud_cnn_model.h5', monitor='val_loss', save_best_only=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=25,
        batch_size=64,
        callbacks=callbacks,
        class_weight={0: 1, 1: len(y_train) // sum(y_train)},  # Alternative to SMOTE
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, framework='tensorflow'):
    if framework == 'tensorflow':
        y_pred_proba = model.predict(X_test).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()
        
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.show()
        
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc='lower left')
        plt.show()
        
        return y_pred, y_pred_proba

def main():
    print("Loading Credit Card Fraud Detection dataset...")
    df = load_data()
    
    print("\nPreprocessing data...")
    X_train, y_train, X_val, y_val, X_test, y_test, scaler = preprocess_data(df)
    
    framework = input("Which framework would you like to use? (pytorch/tensorflow): ").lower()
    
    if framework == 'pytorch':
        print("\nTraining PyTorch CNN model...")
        input_dim = X_train.shape[1]
        model = train_pytorch_model(X_train, y_train, X_val, y_val, input_dim)
        
        print("\nEvaluating PyTorch CNN model...")
        y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, framework='pytorch')
        
    elif framework == 'tensorflow':
        print("\nTraining TensorFlow CNN model...")
        model, history = train_tensorflow_model(X_train, y_train, X_val, y_val)
        
        print("\nEvaluating TensorFlow CNN model...")
        y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, framework='tensorflow')
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    else:
        print("Invalid framework selection. Please choose 'pytorch' or 'tensorflow'.")
        return
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\nROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    
    optimal_idx = np.argmin(np.sqrt((1-tpr)**2 + fpr**2))
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
    
    print("\nClassification Report with Optimal Threshold:")
    print(classification_report(y_test, y_pred_optimal))
    
    cm = confusion_matrix(y_test, y_pred_optimal)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix with Optimal Threshold')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
    
    def predict_transaction(transaction_data, model, scaler, framework='tensorflow', threshold=0.5):
        transaction_data = np.array(transaction_data).reshape(1, -1)
        transaction_data = scaler.transform(transaction_data)
        transaction_data = transaction_data.reshape(1, transaction_data.shape[1], 1)
        
        if framework == 'pytorch':
            model.eval()
            with torch.no_grad():
                transaction_tensor = torch.FloatTensor(transaction_data)
                probability = model(transaction_tensor).item()
        else:  # tensorflow
            probability = model.predict(transaction_data)[0][0]
        
        prediction = 1 if probability >= threshold else 0
        
        return prediction, probability
    
    print("\nExample of real-time fraud detection:")
    
    example_transaction = X_test[0, :, 0]
    pred, prob = predict_transaction(
        example_transaction,
        model,
        scaler,
        framework=framework,
        threshold=optimal_threshold
    )
    
    print(f"Transaction features: {example_transaction[:5]}...")
    print(f"True label: {y_test[0]}")
    print(f"Prediction: {pred} ({'Fraud' if pred == 1 else 'Normal'})")
    print(f"Fraud probability: {prob:.6f}")
    
    return model, scaler, optimal_threshold, framework

if __name__ == "__main__":
    main()