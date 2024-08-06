from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import emoji
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from flask import Flask
from flask_cors import CORS




app = Flask(__name__)
CORS(app) 

# Preprocessing functions
def clean_text(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'#\w+', '', text)  # Remove hashtags
    text = emoji.replace_emoji(text, replace='')  # Remove emojis
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces and trim
    return text

# Load and preprocess data (replace with your actual data source)
df = pd.read_excel(r'C:/Users/Manas Pati Tripathi/text-classifier/htdata.xlsx')
df['text'] = df['text'].apply(clean_text)
y = df['label']

# TF-IDF and ML Models
# Preprocessing
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text']).toarray()
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Logistic Regression:\n", classification_report(y_test, y_pred_lr))

# Support Vector Machine
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("Support Vector Machine:\n", classification_report(y_test, y_pred_svm))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:\n", classification_report(y_test, y_pred_rf))

# Neural Network
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['text'])
X_seq = tokenizer.texts_to_sequences(df['text'])
X_pad = pad_sequences(X_seq, maxlen=100)

X_train_nn, X_test_nn, y_train_nn, y_test_nn = train_test_split(X_pad, y, test_size=0.2, random_state=42)

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(100,)))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train_nn, y_train_nn, epochs=5, batch_size=32, validation_split=0.1)

y_pred_nn = (model.predict(X_test_nn) > 0.5).astype("int32")
print("Neural Network:\n", classification_report(y_test_nn, y_pred_nn))
'''
#BERT MODEL
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)

train_df['text'] = train_df['text'].apply(lambda x: str(x))  # Ensure text is string
test_df['text'] = test_df['text'].apply(lambda x: str(x))  # Ensure text is string

train_encodings = tokenizer(list(train_df['text']), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_df['text']), truncation=True, padding=True, max_length=128)
import torch

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, list(train_df['label']))
test_dataset = TextDataset(test_encodings, list(test_df['label']))
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
)

model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased', num_labels=2)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=lambda p: {"accuracy": accuracy_score(p.label_ids, p.predictions.argmax(-1))}
)
# Train the model
trainer.train()'''
@app.route('/predict', methods=['POST'])
def predict():
    input_text = request.json['text']
    clean_input = clean_text(input_text)

    # TF-IDF
    tfidf_input = vectorizer.transform([clean_input]).toarray()
    lr_pred = lr.predict(tfidf_input)[0]
    svm_pred = svm.predict(tfidf_input)[0]
    rf_pred = rf.predict(tfidf_input)[0]

    # Neural Network
    seq_input = tokenizer.texts_to_sequences([clean_input])
    pad_input = pad_sequences(seq_input, maxlen=100)
    nn_pred = (model.predict(pad_input) > 0.5).astype("int32")[0][0]
    '''
    
    # BERT
    bert_input = tokenize_function([clean_input])
    bert_output = bert_model(torch.tensor(bert_input['input_ids']))
    bert_pred = torch.argmax(bert_output.logits, dim=1).item()'''
    
    return jsonify({
        "logistic_regression": 'Suspicious' if int(lr_pred)==1 else 'Non Suspicious',
        "svm": 'Suspicious' if int(svm_pred)==1 else 'Non Suspicious',
        "random_forest": 'Suspicious' if int(rf_pred)==1 else 'Non Suspicious',
        "neural_network": 'Suspicious' if int(nn_pred)==1 else 'Non Suspicious'})

if __name__ == '__main__':
    app.run(debug=True)
