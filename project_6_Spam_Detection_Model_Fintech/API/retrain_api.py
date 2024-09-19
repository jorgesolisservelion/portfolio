from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import sqlite3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import nltk

# Initialize the FastAPI application
app = FastAPI(
    title="Retrain Models API for Spam detection",
    description="An API for retraining the machine learning models and BERT.",
    version="1.0.0"

    )

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load pre-existing models and vectorizers
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lda_model = joblib.load('lda_model.pkl')
dictionary = joblib.load('dictionary.pkl')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocesses the input text by converting to lowercase, removing punctuation,
    tokenizing, and removing stop words.

    Args:
    text (str): The input text to preprocess.

    Returns:
    str: The preprocessed text.
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    additional_stop_words = {'liking', 'pick', 'quickadded'}
    tokens = [word for word in tokens if word not in stop_words and word not in additional_stop_words]
    return ' '.join(tokens)

# Function to transform text
def transform_text(text):
    """
    Transforms the input text into a feature vector using TF-IDF, LDA, and sentiment analysis.

    Args:
    text (str): The input text to transform.

    Returns:
    np.ndarray: The combined feature vector.
    """
    text_preprocessed = preprocess_text(text)
    tfidf_features = tfidf_vectorizer.transform([text_preprocessed])
    
    text_tokens = text_preprocessed.split()
    text_bow = dictionary.doc2bow(text_tokens)
    lda_features = np.array([dict(lda_model[text_bow]).get(i, 0) for i in range(lda_model.num_topics)]).reshape(1, -1)
    
    sentiment_features = np.array([sentiment_analyzer.polarity_scores(text_preprocessed)[key] for key in ['compound', 'neg', 'neu', 'pos']]).reshape(1, -1)
    
    combined_features = np.hstack([tfidf_features.toarray(), sentiment_features, lda_features])
    return combined_features

# API route for retraining models
@app.post("/retrain")
async def retrain_models(background_tasks: BackgroundTasks):
    """
    Retrain all models with the latest feedback data stored in the feedback database.
    This process is done in the background to avoid blocking the API.

    Args:
    background_tasks (BackgroundTasks): The background task manager.

    Returns:
    dict: A message indicating that retraining has started.
    """
    def retrain():
        # Load the feedback data
        conn = sqlite3.connect('feedback.db')
        c = conn.cursor()
        c.execute('SELECT text, is_spam FROM feedback')
        data = c.fetchall()
        conn.close()

        texts = [row[0] for row in data]
        labels = [row[1] for row in data]

        # Preprocess and transform texts
        features = [transform_text(text) for text in texts]
        features = np.vstack(features)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

        # Retrain traditional models
        models = {
            'random_forest': RandomForestClassifier(),
            'logistic_regression': LogisticRegression(),
            'svm': SVC(probability=True),
            'gradient_boosting': GradientBoostingClassifier()
        }

        for name, model in models.items():
            model.fit(X_train, y_train)
            joblib.dump(model, f'{name}_model.pkl')

        # Retrain BERT model
        class FeedbackDataset(torch.utils.data.Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_len = max_len

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                text = self.texts[idx]
                label = self.labels[idx]

                encoding = self.tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_len,
                    return_token_type_ids=False,
                    pad_to_max_length=True,
                    return_attention_mask=True,
                    return_tensors='pt',
                )

                return {
                    'text': text,
                    'input_ids': encoding['input_ids'].flatten(),
                    'attention_mask': encoding['attention_mask'].flatten(),
                    'labels': torch.tensor(label, dtype=torch.long)
                }

        # Load the BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Prepare the dataset
        max_len = 128
        train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2)
        train_dataset = FeedbackDataset(train_texts, train_labels, tokenizer, max_len)
        val_dataset = FeedbackDataset(val_texts, val_labels, tokenizer, max_len)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=1,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=10,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset
        )

        trainer.train()

        # Save the updated BERT model
        model.save_pretrained('./bert_model')
        tokenizer.save_pretrained('./bert_tokenizer')

    background_tasks.add_task(retrain)
    return {"message": "Retraining started in the background"}

# Test route
@app.get("/")
async def root():
    """
    Welcome message for the Retrain Models API.

    Returns:
    dict: A welcome message.
    """
    return {"message": "Welcome to the Retrain Models API"}
