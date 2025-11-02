import argparse
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import numpy as np
from src.utils import load_papers, build_tfidf
import pandas as pd

def print_topics_nmf(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)
        print(f"Topic {topic_idx+1}: " + ", ".join(top_words))
    return topics

def assign_examples_to_topics(X, nmf_model, papers_df, n_examples=3):
    # Transform documents to topic space
    W = nmf_model.transform(X)
    assigned = W.argmax(axis=1)
    topic_examples = {}
    for t in range(nmf_model.n_components):
        idxs = [i for i,v in enumerate(assigned) if v==t]
        # sort by strength
        idxs_sorted = sorted(idxs, key=lambda i: -W[i,t])[:n_examples]
        topic_examples[t] = papers_df.iloc[idxs_sorted][['title','abstract']].to_dict(orient='records')
    return topic_examples

def main(args):
    df = load_papers(args.data_path)
    docs = df['abstract'].tolist()
    X, vectorizer = build_tfidf(docs, max_features=args.max_features, ngram_range=(1,2))
    feature_names = vectorizer.get_feature_names_out()

    print("Running NMF topic model (this is deterministic for small data)...\n")
    nmf = NMF(n_components=args.n_topics, random_state=42, init='nndsvda', max_iter=500)
    nmf.fit(X)

    print_topics_nmf(nmf, feature_names, args.n_top_words)
    print('\nExample papers for each topic:\n')
    examples = assign_examples_to_topics(X, nmf, df, n_examples=args.n_examples)
    for t, papers in examples.items():
        print(f"--- Topic {t+1} examples ---")
        if not papers:
            print("(no documents assigned)")
        for p in papers:
            print(f"Title: {p['title']}")
            print(f"Abstract: {p['abstract'][:250]}...")
            print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Topic discovery for research papers')
    parser.add_argument('--data_path', type=str, default='data/papers.csv', help='CSV file with title, abstract')
    parser.add_argument('--n_topics', type=int, default=5, help='Number of topics to discover')
    parser.add_argument('--n_top_words', type=int, default=8, help='Top words per topic to display')
    parser.add_argument('--n_examples', type=int, default=3, help='Number of example papers per topic')
    parser.add_argument('--max_features', type=int, default=1000, help='Max TF-IDF features')
    args = parser.parse_args()
    main(args)
