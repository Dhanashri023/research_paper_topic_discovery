# Information Retrieval — Research Paper Topic Discovery
**What this project does**
- Loads a small sample dataset of research paper titles and abstracts.
- Uses TF-IDF + NMF (Non-negative Matrix Factorization) to discover latent topics.
- Prints top words for each topic and shows example papers associated with each topic.

**How to run**
1. (Optional) Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the topic discovery script:
   ```
   python src/main.py --n_topics 5 --n_top_words 8 --n_examples 3
   ```

**Project structure**
- data/papers.csv : sample dataset (title, abstract)
- src/main.py : main CLI script
- src/utils.py : helper functions
- requirements.txt : python dependencies

**Notes**
- This is a simple, educational project meant to be runnable locally.
- For larger real-world datasets, consider more preprocessing, hyperparameter tuning,
  and models like LDA (gensim) or BERTopic (transformer-based).
