from flask import Flask, render_template, request
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from sklearn.decomposition import NMF
from src.utils import load_papers, build_tfidf

app = Flask(__name__, static_folder='static', template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def index():
    df = load_papers('data/papers.csv')
    docs = df['abstract'].tolist()
    X, vectorizer = build_tfidf(docs)

    try:
        n_topics = int(request.form.get('n_topics', 5))
    except:
        n_topics = 5
    n_topics = max(2, min(12, n_topics))
    n_top_words = 8

    nmf = NMF(n_components=n_topics, init='nndsvda', random_state=42, max_iter=500)
    nmf.fit(X)
    feature_names = vectorizer.get_feature_names_out()

    topics_data = []
    W = nmf.transform(X)
    assigned = W.argmax(axis=1)

    for t in range(nmf.n_components):
        topic_words = [feature_names[i] for i in nmf.components_[t].argsort()[:-n_top_words -1:-1]]
        example_idxs = [i for i,v in enumerate(assigned) if v == t][:4]
        examples = df.iloc[example_idxs].to_dict(orient='records')
        topics_data.append({'id': t+1, 'words': topic_words, 'examples': examples})

    return render_template('index.html', topics=topics_data, selected=n_topics)

if __name__ == "__main__":
    app.run(debug=True)
