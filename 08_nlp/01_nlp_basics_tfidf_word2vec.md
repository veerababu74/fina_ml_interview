# NLP Basics: TF-IDF & Word2Vec — Complete Interview Guide

## 🎯 5-Pillar Answer (WHAT / WHY / WHEN / HOW / PITFALLS)
| Pillar | Answer |
|--------|--------|
| **WHAT** | Converting text into numerical representations for ML models |
| **WHY** | Machines can't process raw text; need vectors capturing meaning |
| **WHEN** | Classification, clustering, search, recommendation on text data |
| **HOW** | Preprocessing → Vectorization (TF-IDF or Word2Vec) → Model |
| **PITFALLS** | Ignoring stopwords, not handling OOV words, wrong tokenization |

## Core Concept
NLP transforms unstructured text into structured numerical features. Key representations:
- **Bag of Words (BoW)**: word counts, ignores order
- **TF-IDF**: weighted word importance across documents
- **Word2Vec**: dense semantic embeddings

## Text Preprocessing Pipeline

```python
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Complete text preprocessing pipeline."""
    # Lowercase
    text = text.lower()
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove special characters, keep only letters/spaces
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# Example
texts = [
    "Machine learning is a subset of artificial intelligence.",
    "Deep learning uses neural networks with many layers.",
    "Natural language processing handles text and speech data."
]
processed = [preprocess_text(t) for t in texts]
print(processed)
# ['machine learning subset artificial intelligence',
#  'deep learning use neural network many layer',
#  'natural language processing handle text speech datum']
```

## TF-IDF

### Formula
- **TF(t, d)** = count of term t in document d / total terms in d
- **IDF(t)** = log(N / df(t)) where N = total docs, df(t) = docs containing t
- **TF-IDF(t, d)** = TF(t, d) × IDF(t)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np

# Sample dataset
texts = [
    "I love this movie, it is fantastic",
    "Terrible film, waste of time",
    "Great acting and brilliant story",
    "Worst movie ever, completely awful",
    "Amazing cinematography and direction",
    "Boring and predictable plot"
]
labels = [1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

# TF-IDF + Logistic Regression pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),      # unigrams and bigrams
        min_df=2,                 # ignore rare terms
        max_df=0.95,              # ignore very common terms
        sublinear_tf=True         # apply log normalization to TF
    )),
    ('clf', LogisticRegression(random_state=42, max_iter=1000))
])

pipeline.fit(texts, labels)

# Show top TF-IDF features
tfidf = pipeline.named_steps['tfidf']
feature_names = tfidf.get_feature_names_out()
print("Vocabulary size:", len(feature_names))

# Predict new text
new_texts = ["This is an excellent film!", "Absolutely dreadful experience"]
predictions = pipeline.predict(new_texts)
probabilities = pipeline.predict_proba(new_texts)
print("Predictions:", predictions)          # [1, 0]
print("Probabilities:", probabilities.round(2))
```

## Word2Vec

### CBOW vs Skip-gram
| Aspect | CBOW | Skip-gram |
|--------|------|-----------|
| **Task** | Predict center word from context | Predict context from center word |
| **Training speed** | Faster | Slower |
| **Rare words** | Worse | Better |
| **Common words** | Better | Good |
| **Use case** | Large corpus | Small corpus, rare words |

```python
from gensim.models import Word2Vec
import numpy as np

# Train Word2Vec
sentences = [text.split() for text in processed]
# Add more sentences for a real model
sentences += [
    ["neural", "network", "deep", "learning"],
    ["decision", "tree", "random", "forest"],
    ["support", "vector", "machine", "kernel"],
    ["logistic", "regression", "classification"],
    ["natural", "language", "processing", "text"]
]

# Skip-gram model (sg=1)
model = Word2Vec(
    sentences,
    vector_size=100,   # embedding dimensions
    window=5,          # context window size
    min_count=1,       # minimum word frequency
    sg=1,              # 1=Skip-gram, 0=CBOW
    workers=4,
    seed=42
)

# Most similar words
# model.wv.most_similar('learning', topn=5)

# Word arithmetic: king - man + woman ≈ queen
# model.wv.most_similar(positive=['king', 'woman'], negative=['man'])

# Get word vector
vector = model.wv['learning']
print("Vector shape:", vector.shape)  # (100,)

# Document embedding: average word vectors
def doc_to_vec(tokens, model, vec_size=100):
    """Convert document to vector by averaging word embeddings."""
    vectors = []
    for token in tokens:
        if token in model.wv:
            vectors.append(model.wv[token])
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(vec_size)

doc_vecs = np.array([doc_to_vec(s, model) for s in sentences])
print("Document matrix shape:", doc_vecs.shape)
```

## Sentence Embeddings with sentence-transformers

```python
# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model_st = SentenceTransformer('all-MiniLM-L6-v2')

sentences_demo = [
    "Machine learning automates pattern recognition",
    "Deep learning is a subset of machine learning",
    "Python is a programming language",
    "Pandas is used for data manipulation"
]

# Encode sentences
embeddings = model_st.encode(sentences_demo)
print("Embeddings shape:", embeddings.shape)  # (4, 384)

# Compute cosine similarity
sim_matrix = cosine_similarity(embeddings)
print("Similarity between sent 0 and 1:", sim_matrix[0][1].round(3))
print("Similarity between sent 0 and 2:", sim_matrix[0][2].round(3))
# ML sentences should be more similar to each other
```

## Complete Text Classification Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
import numpy as np

np.random.seed(42)

# Simulate larger dataset
positive_texts = [
    "excellent product highly recommend",
    "amazing quality very satisfied",
    "best purchase ever great value",
    "fantastic experience will buy again",
    "outstanding performance exceeded expectations",
    "wonderful service and quality product",
]
negative_texts = [
    "terrible quality do not buy",
    "worst product ever complete waste",
    "very disappointed poor quality",
    "horrible experience never again",
    "broken on arrival total disappointment",
    "awful product poor customer service",
]

texts = positive_texts + negative_texts
labels = [1]*len(positive_texts) + [0]*len(negative_texts)

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.3, random_state=42, stratify=labels
)

# Full pipeline
clf_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=1
    )),
    ('clf', LogisticRegression(
        C=1.0,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    ))
])

clf_pipeline.fit(X_train, y_train)
y_pred = clf_pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Interview Questions & Answers

**Q1: What is TF-IDF and why use it over simple word counts?**
A: TF-IDF (Term Frequency-Inverse Document Frequency) weights terms by their importance. Simple counts give high weight to common words like "the". TF-IDF downweights words common across all documents (via IDF) while upweighting document-specific terms, giving better signal.

**Q2: What is the difference between CBOW and Skip-gram in Word2Vec?**
A: CBOW predicts a center word from its context words (faster, good for frequent words). Skip-gram predicts context words from a center word (slower but better for rare words and small datasets). Skip-gram is generally preferred for smaller datasets.

**Q3: How do you handle out-of-vocabulary (OOV) words in Word2Vec?**
A: Options include: (1) use FastText which generates subword embeddings so OOV words get embeddings from character n-grams, (2) use a special UNK token, (3) use zero vector or random initialization, (4) use sentence transformers (BERT-based) which handle OOV via subword tokenization.

**Q4: When would you choose TF-IDF over Word2Vec?**
A: Use TF-IDF when: dataset is small, interpretability matters, computational resources are limited, task involves keyword matching. Use Word2Vec/embeddings when: semantic similarity is important, you have large datasets, you need to capture context.

**Q5: What is the cosine similarity and why is it preferred for text?**
A: Cosine similarity measures the angle between two vectors: sim(A,B) = (A·B)/(|A||B|). It's preferred for text because it's invariant to document length (unlike dot product), so a short and long document with similar topics have high similarity.

**Q6: What is n-gram and why use bigrams?**
A: An n-gram is a sequence of n words. "not good" as a bigram captures negation better than individual words "not" and "good". Bigrams and trigrams preserve some word order context that pure BoW misses.

## Common Pitfalls
1. **Not cleaning text**: HTML tags, special characters, URLs pollute features
2. **Wrong tokenization**: splitting on spaces misses "don't" → "don" and "t"
3. **Stopword removal blindly**: domain stopwords differ (removing "not" kills sentiment)
4. **TF-IDF on very short texts**: IDF becomes meaningless with few documents
5. **Word2Vec with tiny corpus**: needs at least 10K sentences for quality embeddings
6. **Ignoring class imbalance**: use class_weight='balanced' in classifier

## Quick Reference Cheat Sheet
```
TF-IDF:
  tfidf = TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True)

Word2Vec (gensim):
  model = Word2Vec(sentences, vector_size=300, window=5, min_count=5, sg=1)
  
Sentence Embeddings:
  model = SentenceTransformer('all-MiniLM-L6-v2')
  embeddings = model.encode(sentences)

TF-IDF > Word2Vec: small data, interpretability, keyword tasks
Word2Vec > TF-IDF: semantic similarity, large data, analogies
FastText: handles OOV via subwords
```
