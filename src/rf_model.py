import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('creditcard.csv')
data_rf = df.copy()

# Feature Engineering: Convert 'Time' (seconds) to hours of the day (0-24)
# np.ceil rounds up, % 86400 gets seconds within the current day, / 3600 converts to hours
data_rf['scaled_time'] = data_rf['Time'].apply(lambda x: np.ceil(float(x) % 86400) / 3600)

# Scale the 'Amount' column using RobustScaler
# RobustScaler is less prone to outliers compared to StandardScaler
data_rf['scaled_amount'] = RobustScaler().fit_transform(data_rf['Amount'].values.reshape(-1,1))

# Drop the original raw columns as we now have scaled versions
data_rf.drop(['Time', 'Amount'], axis=1, inplace=True)

# Define features (X) and target variable (y)
X = data_rf.drop('Class', axis=1)
y = data_rf['Class']

# Print feature details
print(f"Training with all features. Number of features: {X.shape[1]}")
print(f"Feature list: {X.columns.tolist()}")

# Split data into training and testing sets (80% train, 20% test)
# 'stratify=y' ensures the proportion of classes is the same in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Print dataset shapes
print(f"Training set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(
    n_estimators=200,       # Number of trees in the forest
    max_depth=15,           # Maximum depth of the tree to prevent overfitting
    random_state=42,        # Seed for reproducibility
    class_weight='balanced', # Adjust weights inversely proportional to class frequencies (good for imbalanced data)
    n_jobs=-1               # Use all available processor cores
)

# Train the model
print("Training Random Forest model...")
rf_model.fit(X_train, y_train)

# Predict probabilities for the positive class (column index 1)
y_rf_scores = rf_model.predict_proba(X_test)[:, 1]

# Calculate Precision and Recall values for the curve
precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_rf_scores)
# Calculate Area Under the Precision-Recall Curve (AUPRC)
auprc_rf = auc(recall_rf, precision_rf)

# Plotting the Precision-Recall Curve
plt.figure(figsize=(10, 6))
plt.plot(recall_rf, precision_rf, color='green', label=f'Random Forest (AUPRC = {auprc_rf:.4f})')

plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve: Random Forest', fontsize=16)
plt.legend()
plt.grid(True)
plt.show()

# Generate binary predictions and print the final score
y_rf_pred = rf_model.predict(X_test)

# Print the AUPRC score
print(f"\nRandom Forest AUPRC Score: {auprc_rf:.8f}")
