from flask import Flask, render_template, request
import pandas as pd
from sklearn.decomposition import NMF
from src.utils import load_papers, build_tfidf

app = Flask(__name__)

df = load_papers("data/papers.csv")
docs = df["abstract"].tolist()
X, vectorizer = build_tfidf(docs)
feature_names = vectorizer.get_feature_names_out()

@app.route("/", methods=["GET", "POST"])
def index():
    topics_output = []
    topic_examples = {}
    if request.method == "POST":
        n_topics = int(request.form.get("n_topics", 5))
        n_top_words = int(request.form.get("n_top_words", 8))
        nmf = NMF(n_components=n_topics, random_state=42, init="nndsvda", max_iter=500)
        W = nmf.fit_transform(X)
        H = nmf.components_
        assignments = W.argmax(axis=1)

        for i, comp in enumerate(H):
            top_words = [feature_names[j] for j in comp.argsort()[:-n_top_words - 1:-1]]
            topics_output.append({"topic": i+1, "words": top_words})

            idxs = [k for k,v in enumerate(assignments) if v==i][:3]
            topic_examples[i] = df.iloc[idxs][["title","abstract"]].to_dict(orient="records")

        return render_template("index.html", topics=topics_output, examples=topic_examples)

    return render_template("index.html", topics=None, examples=None)

if __name__ == "__main__":
    app.run(debug=True)
