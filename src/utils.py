import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

def load_papers(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=['abstract'])
    return df

def simple_preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def build_tfidf(docs, max_features=2000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=1, max_features=max_features,
                                 stop_words='english', ngram_range=ngram_range,
                                 preprocessor=simple_preprocess)
    X = vectorizer.fit_transform(docs)
    return X, vectorizer
