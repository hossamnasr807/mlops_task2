import os
import json
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load preprocessed data
train_df = pd.read_csv('data/train.csv')
X_train = train_df.drop('class', axis=1)
y_train = train_df['class']

# Train model
model = DecisionTreeClassifier(max_depth=None, random_state=42)
model.fit(X_train, y_train)

# Ensure model directory exists
os.makedirs('models', exist_ok=True)

# Save model
joblib.dump(model, 'models/model.pkl')
print("Saved model to models/model.pkl")

# Save training metrics
train_acc = accuracy_score(y_train, model.predict(X_train))
with open('train_metrics.json', 'w') as f:
    json.dump({'train_accuracy': train_acc}, f)
print("Saved train_metrics.json")
