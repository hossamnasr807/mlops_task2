import pandas as pd
import joblib
import json
import os


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


# Load training data
df = pd.read_csv('data/train.csv')
X = df.drop(columns=['class'])
y = df['class']


clf = LogisticRegression(max_iter=2000, random_state=42, multi_class='multinomial', solver='saga')
clf.fit(X, y)

# ensure models dir
os.makedirs('models', exist_ok=True)
model_path = 'models/model.pkl'
print('Saved model to', model_path)


# quick-train metrics on train set
train_preds = clf.predict(X)
acc = (train_preds == y).mean()
metrics = {'train_accuracy': float(acc), 'model': 'logistic'}
with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
print('Saved metrics.json')