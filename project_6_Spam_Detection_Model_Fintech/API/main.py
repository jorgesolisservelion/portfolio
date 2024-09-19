from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import sqlite3

# Initialize the FastAPI application
app = FastAPI(
    title="Spam Detection API",
    description="An API to classify text as SPAM or NOT SPAM using traditional machine learning models and BERT.",
    version="1.0.0"
)

# Define threshold for spam detection
THRESHOLD = 0.5

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')

# Load trained models and vectorizers
models_dict = {
    'random_forest': joblib.load('random_forest_model.pkl'),
    'logistic_regression': joblib.load('logistic_regression_model.pkl'),
    'svm': joblib.load('svm_model.pkl'),
    'gradient_boosting': joblib.load('gradient_boosting_model.pkl')
}

tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
lda_model = joblib.load('lda_model.pkl')
dictionary = joblib.load('dictionary.pkl')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# Initialize the sentiment analyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

# Initialize SQLite database
conn = sqlite3.connect('feedback.db')
c = conn.cursor()
c.execute('''
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY,
    text TEXT,
    is_spam INTEGER
)
''')
conn.commit()

# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess the input text by lowering case, removing punctuation, tokenizing, 
    and removing stop words and additional specific words.
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
    Transform the input text to extract features using TF-IDF, LDA, and sentiment analysis.
    """
    text_preprocessed = preprocess_text(text)
    tfidf_features = tfidf_vectorizer.transform([text_preprocessed])
    
    text_tokens = text_preprocessed.split()
    text_bow = dictionary.doc2bow(text_tokens)
    lda_features = np.array([dict(lda_model[text_bow]).get(i, 0) for i in range(lda_model.num_topics)]).reshape(1, -1)
    
    sentiment_features = np.array([sentiment_analyzer.polarity_scores(text_preprocessed)[key] for key in ['compound', 'neg', 'neu', 'pos']]).reshape(1, -1)
    
    combined_features = np.hstack([tfidf_features.toarray(), sentiment_features, lda_features])
    return combined_features

# Function to predict SPAM with traditional models
def predict_spam(text, model_name):
    """
    Predict if the input text is SPAM or NOT SPAM using traditional machine learning models.
    """
    if model_name not in models_dict:
        raise HTTPException(status_code=400, detail="Model not found")
    
    model = models_dict[model_name]
    features = transform_text(text)
    prediction = model.predict(features)
    probability = model.predict_proba(features)
    return prediction[0], probability[0][1]

# Function to predict SPAM with BERT
def predict_spam_bert(text, model, tokenizer, max_len, device, threshold=THRESHOLD):
    """
    Predict if the input text is SPAM or NOT SPAM using a BERT model.
    """
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=False,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt',
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    model.eval()  # Set the model to evaluation mode

    with torch.no_grad():  # Disable gradient calculation for evaluation
        outputs = model(input_ids, attention_mask=attention_mask)
        probabilities = torch.softmax(outputs.logits, dim=1)
        spam_prob = probabilities[0][1].item()  # Probability of the SPAM class

    is_spam = spam_prob >= threshold
    
    return is_spam, spam_prob

# Data model for the prediction request
class TextRequest(BaseModel):
    text: str = Field(..., example="This is an example text to evaluate if it is SPAM or not.")

# Data model for feedback
class FeedbackRequest(BaseModel):
    text: str = Field(..., example="This is the text that was misclassified.")
    is_spam: bool = Field(..., example=True)

# API route for prediction
@app.post("/predict", summary="Predict if text is SPAM or NOT SPAM", response_description="Prediction results from all models", responses={
    200: {
        "description": "Prediction results from all models",
        "content": {
            "application/json": {
                "example": {
                    "random_forest": {"is_spam": True, "probability": 0.85},
                    "logistic_regression": {"is_spam": False, "probability": 0.45},
                    "svm": {"is_spam": True, "probability": 0.75},
                    "gradient_boosting": {"is_spam": False, "probability": 0.35},
                    "bert": {"is_spam": True, "probability": 0.65}
                }
            }
        }
    },
    422: {
        "description": "Validation error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    }
})
async def predict(request: TextRequest):
    """
    Predict if the input text is SPAM or NOT SPAM using both traditional machine learning models and a BERT model.

    - **text**: The input text to classify
    - Returns: A dictionary with prediction results from each model
    """
    results = {}
    for model_name in models_dict.keys():
        prediction, probability = predict_spam(request.text, model_name)
        results[model_name] = {
            "is_spam": bool(prediction),
            "probability": probability
        }
    
    # BERT model prediction
    is_spam_bert, prob_bert = predict_spam_bert(request.text, bert_model, tokenizer, 128, device, THRESHOLD)
    results['bert'] = {
        "is_spam": bool(is_spam_bert),
        "probability": prob_bert
    }
    
    return results

# API route for feedback
@app.post("/feedback", summary="Provide feedback on the prediction", response_description="Feedback received confirmation", responses={
    200: {
        "description": "Feedback successfully received",
        "content": {
            "application/json": {
                "example": {"message": "Feedback received"}
            }
        }
    },
    422: {
        "description": "Validation error",
        "content": {
            "application/json": {
                "example": {
                    "detail": [
                        {
                            "loc": ["body", "text"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        },
                        {
                            "loc": ["body", "is_spam"],
                            "msg": "field required",
                            "type": "value_error.missing"
                        }
                    ]
                }
            }
        }
    }
})
async def feedback(request: FeedbackRequest):
    """
    Provide feedback on the prediction.

    - **text**: The text that was misclassified
    - **is_spam**: The correct classification (True if SPAM, False otherwise)
    """
    c.execute('''
    INSERT INTO feedback (text, is_spam) VALUES (?, ?)
    ''', (request.text, int(request.is_spam)))
    conn.commit()
    return {"message": "Feedback received"}

# Test route
@app.get("/", summary="Root endpoint", response_description="Welcome message")
async def root():
    """
    Root endpoint to test if the API is running.

    - Returns: A welcome message
    """
    return {"message": "Hello World"}

