import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# load model and test
model = joblib.load('models/model.pkl')
print('Loaded model')

df = pd.read_csv('data/test.csv')
X = df.drop(columns=['class'])
y = df['class']

preds = model.predict(X)
acc = accuracy_score(y, preds)
cm = confusion_matrix(y, preds, labels=np.unique(y))  # use sorted labels

metrics = {'test_accuracy': float(acc), 'model': getattr(model, '__class__', 'model').__name__}
metrics['classification_report'] = classification_report(y, preds, output_dict=True)

with open('metrics.json', 'w') as f:
    json.dump(metrics, f)
print('Saved metrics.json')

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print('Saved confusion_matrix.png')
