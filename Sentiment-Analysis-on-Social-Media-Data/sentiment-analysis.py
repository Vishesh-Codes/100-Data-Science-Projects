import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader, TensorDataset
import torch
from tqdm import tqdm
import nltk
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download("punkt")

# Load your dataset (replace 'your_dataset.csv' with the actual filename)
try:
    df = pd.read_csv('https://raw.githubusercontent.com/Vishesh-Codes/100-Data-Science-Projects/main/Sentiment-Analysis-on-Social-Media-Data/sentimentdataset.csv')
except Exception as e:
    st.error(f"Error loading the dataset: {e}")

# Sample: Assuming your dataset has 'text' and 'label' columns
X = df['text']
y = df['sentiment']

# Encode labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define a function for text preprocessing
def preprocess_text(text):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    ps = nltk.PorterStemmer()
    tokens = nltk.word_tokenize(text)
    tokens = [ps.stem(token) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return ' '.join(tokens)

# Apply text preprocessing to the training and testing sets
X_train = X_train.apply(preprocess_text)
X_test = X_test.apply(preprocess_text)

# Train a TF-IDF vectorizer and a logistic regression model
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
logistic_regression_model = LogisticRegression()

# Create a pipeline for easy implementation
pipeline = make_pipeline(tfidf_vectorizer, logistic_regression_model)

# Train the model
pipeline.fit(X_train, y_train)

# ----- BERT Model for Transfer Learning -----

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Tokenize and encode the training set
X_train_tokens = tokenizer(X_train.tolist(), padding=True, truncation=True, return_tensors='pt')
y_train_tensor = torch.tensor(y_train)

# Tokenize and encode the test set
X_test_tokens = tokenizer(X_test.tolist(), padding=True, truncation=True, return_tensors='pt')
y_test_tensor = torch.tensor(y_test)

# Define a DataLoader for training
train_dataset = TensorDataset(X_train_tokens['input_ids'], X_train_tokens['attention_mask'], y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

# Train the BERT model
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move the model to the appropriate device
bert_model.to(device)

# Train the model
num_epochs = 3
for epoch in range(num_epochs):
    bert_model.train()
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = bert_model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Evaluate the BERT model on the test set
bert_model.eval()
with torch.no_grad():
    outputs = bert_model(**X_test_tokens, labels=y_test_tensor)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=1).cpu().numpy()

# Evaluate the BERT model
accuracy_bert = accuracy_score(y_test, predictions)

# ----- Random Forest Model -----

# Tokenize and encode the entire dataset using TF-IDF
X_tfidf = tfidf_vectorizer.transform(X)

# Train a Random Forest classifier
random_forest_model = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest_model.fit(X_tfidf, y_encoded)

# Make predictions on the test set
y_pred_rf = random_forest_model.predict(tfidf_vectorizer.transform(X_test))

# Evaluate the Random Forest model
accuracy_rf = accuracy_score(y_test, y_pred_rf)

# Streamlit app
st.title("Sentiment Analysis App")

# User Input
user_input = st.text_area("Enter a sentence for sentiment analysis:")

# Predictions
if user_input:
    # Preprocess user input
    user_input_processed = preprocess_text(user_input)

    # Logistic Regression Model Prediction
    lr_prediction = pipeline.predict([user_input_processed])[0]

    # BERT Model Prediction
    user_input_tokens = tokenizer([user_input_processed], padding=True, truncation=True, return_tensors='pt')
    user_input_tokens = {key: val.to(device) for key, val in user_input_tokens.items()}
    with torch.no_grad():
        bert_output = bert_model(**user_input_tokens)
        bert_prediction = torch.argmax(bert_output.logits, dim=1).item()

    # Random Forest Model Prediction
    rf_prediction = label_encoder.inverse_transform(random_forest_model.predict(tfidf_vectorizer.transform([user_input_processed])))[0]

    # Display Predictions
    st.header("Predictions:")
    st.write(f"Logistic Regression Model Prediction: {label_encoder.inverse_transform([lr_prediction])[0]}")
    st.write(f"BERT Model Prediction: {label_encoder.inverse_transform([bert_prediction])[0]}")
    st.write(f"Random Forest Model Prediction: {rf_prediction}")

# Display Model Accuracy
st.header("Model Accuracy:")
st.write(f"Logistic Regression Model Accuracy: {accuracy_score(y_test, pipeline.predict(X_test)):.2f}")
st.write(f"BERT Model Accuracy: {accuracy_bert:.2f}")
st.write(f"Random Forest Model Accuracy: {accuracy_rf:.2f}")
