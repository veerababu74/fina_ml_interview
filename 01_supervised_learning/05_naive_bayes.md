# 🎲 Naive Bayes — Complete Interview Guide

> **5-Pillar Answer Framework:** WHAT | WHY | WHEN | HOW | PITFALLS

---

## 🏛️ The 5-Pillar Answer

| Pillar | Answer |
|--------|--------|
| **WHAT** | A probabilistic classifier using Bayes' theorem with a "naive" conditional independence assumption among features |
| **WHY** | Extremely fast (O(nd) training), requires little data, handles high-dimensional data well, natural for text, gives calibrated probabilities |
| **WHEN** | Text classification (spam, sentiment), document categorization, real-time prediction, small training data, high-dimensional discrete features |
| **HOW** | Estimates P(x_j\|y) for each feature/class pair; prediction = argmax P(y)·Π P(xⱼ\|y); Laplace smoothing for zero frequencies |
| **PITFALLS** | Independence assumption often violated (correlated features → overconfident), poor with continuous correlated features, zero frequency problem |

---

## 📐 Bayes' Theorem Derivation

### Bayes' Theorem
```
P(y|x) = P(x|y) · P(y) / P(x)
```
- P(y|x): **Posterior** — probability of class y given features x
- P(x|y): **Likelihood** — probability of features x given class y
- P(y): **Prior** — probability of class y (base rate)
- P(x): **Evidence** — marginal probability of x (constant for all classes)

### Naive Bayes Assumption

Assume features are **conditionally independent** given the class:
```
P(x₁, x₂, ..., xₚ | y) = Π_{j=1}^{p} P(xⱼ | y)
```

This is the "naive" part — rarely true but surprisingly effective!

### Classification Rule

```
ŷ = argmax_y P(y) · Π_{j=1}^{p} P(xⱼ | y)

  = argmax_y [log P(y) + Σ_{j=1}^{p} log P(xⱼ | y)]  ← log-sum for numerical stability
```

No need to compute P(x) since it's constant across classes.

---

## 🔢 Three Variants

### 1. Gaussian Naive Bayes (continuous features)

Assumes features follow Gaussian distribution within each class:
```
P(xⱼ | y=c) = (1/√(2π σ²_{jc})) · exp(-(xⱼ - μ_{jc})² / (2σ²_{jc}))
```
Parameters estimated from data: μ_{jc} = mean of feature j in class c, σ²_{jc} = variance.

### 2. Multinomial Naive Bayes (count data — text)

For feature counts (word counts, TF-IDF):
```
P(xⱼ | y=c) = (count(xⱼ, c) + α) / (Σₖ count(xₖ, c) + α·V)
```
Where α is the Laplace smoothing parameter, V is vocabulary size.

### 3. Bernoulli Naive Bayes (binary features)

For binary features (word present/absent):
```
P(xⱼ | y=c) = p_{jc}^{xⱼ} · (1 - p_{jc})^{1-xⱼ}
```
Where p_{jc} = P(xⱼ=1 | y=c). Explicitly models absence of features.

---

## 🔢 Zero Frequency Problem & Laplace Smoothing

### The Problem
If a word "amazing" never appeared in spam emails during training:
```
P("amazing" | spam) = 0/total = 0
```
Then P(spam | document with "amazing") = 0 regardless of other evidence!

### Laplace (Additive) Smoothing
Add α > 0 pseudo-counts to every feature/class combination:
```
P(xⱼ=v | y=c) = (count(xⱼ=v, y=c) + α) / (count(y=c) + α·|V|)
```
- α = 1: Laplace smoothing (uniform prior)
- 0 < α < 1: Lidstone smoothing
- Ensures no zero probabilities

---

## 🤔 Why Does NB Work Despite "Naive" Assumption?

1. **Zero-one loss robustness**: For classification, we only need argmax, not accurate probability magnitudes
2. **Correlated features cancel out**: If features x₁ and x₂ are perfectly correlated, they double-count but don't change the argmax
3. **Works well with high-dimensional data**: With many features, the independence assumption reduces variance dramatically
4. **Empirical evidence**: On text tasks, NB often matches logistic regression performance with much less data

---

## 💻 Full Implementation with Text Classification

```python
import numpy as np
import pandas as pd
from sklearn.naive_bayes import (
    GaussianNB, MultinomialNB, BernoulliNB, ComplementNB
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import (
    cross_val_score, GridSearchCV, train_test_split
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, accuracy_score
)
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification, fetch_20newsgroups
import warnings
warnings.filterwarnings('ignore')


# ══════════════════════════════════════════════════════════════════
# PART 1: GAUSSIAN NAIVE BAYES (continuous features)
# ══════════════════════════════════════════════════════════════════

print("=" * 55)
print("GAUSSIAN NAIVE BAYES")
print("=" * 55)

X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=6,
    n_classes=3, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# GaussianNB doesn't need feature scaling (models each class separately)
gnb = GaussianNB(
    var_smoothing=1e-9   # Adds to variance to prevent numerical issues
)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
y_proba = gnb.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"ROC-AUC (macro OvR): {roc_auc_score(y_test, y_proba, multi_class='ovr'):.4f}")

# Inspect learned parameters
print(f"\nClass priors: {gnb.class_prior_}")
print(f"Class means (first 3 features):\n{gnb.theta_[:, :3]}")


# ══════════════════════════════════════════════════════════════════
# PART 2: TEXT CLASSIFICATION — SPAM DETECTION EXAMPLE
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("MULTINOMIAL NAIVE BAYES — TEXT CLASSIFICATION")
print("=" * 55)

# Synthetic spam/ham dataset
texts = [
    "buy cheap pills online now free offer",
    "click here to win million dollars prize",
    "free credit card apply now limited offer",
    "urgent you have won lottery claim prize",
    "meeting tomorrow at 10am in conference room",
    "please review the attached project report",
    "quarterly sales report for your review",
    "can we schedule a call to discuss results",
    "hello how are you doing today",
    "reminder team standup at 3pm this afternoon",
    "best price guarantee buy now free shipping",
    "discount offer exclusive deal expires tonight",
    "project deadline is next friday please confirm",
    "thanks for your email I will respond soon",
    "win free vacation tropical paradise click here"
]
labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]
# 1 = spam, 0 = ham

# ── Pipeline: TF-IDF + Multinomial NB ────────────────────────────
mnb_pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),    # Unigrams + bigrams
        min_df=1,              # Minimum document frequency
        sublinear_tf=True,     # Log(tf+1) instead of tf
        stop_words='english'
    )),
    ('classifier', MultinomialNB(
        alpha=1.0,             # Laplace smoothing (0 = no smoothing)
        fit_prior=True         # Learn class priors from data
    ))
])

mnb_pipeline.fit(texts, labels)

# Predictions
test_messages = [
    "free offer buy now click",
    "meeting agenda for next week",
    "win prize click now free",
    "quarterly report review"
]

for msg in test_messages:
    pred = mnb_pipeline.predict([msg])[0]
    prob = mnb_pipeline.predict_proba([msg])[0]
    label = "SPAM" if pred == 1 else "HAM"
    print(f"{label} ({prob[1]:.2f} spam prob): '{msg}'")


# ── Top Spam / Ham Words ──────────────────────────────────────────
vectorizer = mnb_pipeline.named_steps['vectorizer']
nb_model = mnb_pipeline.named_steps['classifier']
feature_names = vectorizer.get_feature_names_out()

# Log probability difference: how much more likely in spam vs ham
log_prob_diff = nb_model.feature_log_prob_[1] - nb_model.feature_log_prob_[0]
top_spam_words = [feature_names[i] for i in np.argsort(log_prob_diff)[-10:]]
top_ham_words = [feature_names[i] for i in np.argsort(log_prob_diff)[:10]]
print(f"\nTop spam words: {top_spam_words}")
print(f"Top ham words:  {top_ham_words}")


# ══════════════════════════════════════════════════════════════════
# PART 3: BERNOULLI NAIVE BAYES (binary features)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("BERNOULLI NAIVE BAYES — BINARY FEATURES")
print("=" * 55)

bnb_pipeline = Pipeline([
    ('vectorizer', CountVectorizer(
        max_features=5000,
        ngram_range=(1, 1),
        binary=True,           # Convert to binary (presence/absence)
        stop_words='english'
    )),
    ('classifier', BernoulliNB(
        alpha=1.0,             # Laplace smoothing
        binarize=None          # Features already binary after CountVectorizer
    ))
])

bnb_pipeline.fit(texts, labels)
bnb_pred = bnb_pipeline.predict(test_messages)
print(f"BernoulliNB predictions: {['SPAM' if p else 'HAM' for p in bnb_pred]}")


# ══════════════════════════════════════════════════════════════════
# PART 4: TUNING ALPHA (LAPLACE SMOOTHING)
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 55)
print("TUNING LAPLACE SMOOTHING ALPHA")
print("=" * 55)

# Larger dataset for CV
categories = ['sci.space', 'talk.politics.guns', 'rec.sport.hockey']
newsgroups = fetch_20newsgroups(
    subset='train', categories=categories, remove=('headers', 'footers', 'quotes')
)

alpha_values = [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
cv_scores = []

for alpha in alpha_values:
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, sublinear_tf=True)),
        ('nb', MultinomialNB(alpha=alpha))
    ])
    scores = cross_val_score(pipeline, newsgroups.data, newsgroups.target,
                              cv=3, scoring='accuracy')
    cv_scores.append(scores.mean())
    print(f"alpha={alpha:8.4f} → CV Accuracy: {scores.mean():.4f}")

best_alpha = alpha_values[np.argmax(cv_scores)]
print(f"\nBest alpha: {best_alpha}")
```

---

## ❓ Interview Questions & Answers

### Q1: Why is Naive Bayes "naive" and when does it still work?
**Answer:** "Naive" refers to the conditional independence assumption: P(x₁,...,xₚ|y) = Π P(xⱼ|y). This is almost never true (e.g., words "New" and "York" are correlated). It still works because: (1) For classification, we only need the correct argmax, not accurate probabilities, (2) In high dimensions, reduced variance from independence assumption outweighs bias, (3) Correlated features vote similarly — equivalent to upweighting that evidence.

### Q2: How does Laplace smoothing help and what value of alpha should you use?
**Answer:** Without smoothing, any feature-class combination unseen in training gets probability 0, making the entire posterior 0. Laplace smoothing adds α pseudo-counts: P(xⱼ|y) = (count + α)/(total + α·V). α=1 is the classic Laplace prior (uniform prior over vocabulary). For large vocabularies, α=0.1 often works better. Tune α as a hyperparameter via cross-validation — it's directly the regularization strength.

### Q3: What's the difference between Multinomial and Bernoulli NB for text?
**Answer:** **MultinomialNB** uses word counts (how many times each word appears) — better for longer documents where frequency matters. **BernoulliNB** uses word presence/absence (binary) — explicitly models missing words, better for short texts. BernoulliNB penalizes missing words (subtracts log(1-p) for absent features), MultinomialNB ignores absent words. For spam detection with short messages, BernoulliNB often performs better.

### Q4: When would you choose Naive Bayes over Logistic Regression?
**Answer:** Prefer NB when: (1) **Very limited training data** — NB's strong prior (independence) helps avoid overfitting, (2) **Real-time requirements** — NB is O(1) prediction, (3) **Online learning** — easy to update with new data incrementally, (4) **High-dimensional discrete data** (text) — scales well, (5) **Class probability estimates matter** (well-calibrated). Prefer Logistic Regression when: you have sufficient data, features are correlated, accuracy is the priority.

### Q5: What is ComplementNB and when is it better than MultinomialNB?
**Answer:** ComplementNB estimates parameters using the complement of each class (all other classes), then uses 1/weight as the classification weight. It corrects for class imbalance and is more robust when the independence assumption is violated. Empirically, ComplementNB often outperforms MultinomialNB on imbalanced text classification tasks (e.g., multi-class newsgroups). Use `ComplementNB` from sklearn as a drop-in replacement when classes are imbalanced.

---

## ⚠️ Common Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Zero frequency without smoothing | P(class) = 0 for any unseen feature | Always set alpha > 0 |
| Using GaussianNB for text counts | Counts can't be negative (Gaussian allows) | Use MultinomialNB or BernoulliNB |
| Highly correlated features | Overconfident probabilities | Feature selection, or use LogReg |
| Not normalizing for MultinomialNB | Needs non-negative features | Use CountVectorizer or TF-IDF |
| Treating ordinal as Gaussian | Violates distribution assumption | Use appropriate variant |
| Ignoring class imbalance | Biased prior dominates | Set class_prior manually or use ComplementNB |

---

## 📋 Quick Reference Cheat Sheet

```
Naive Bayes Quick Reference
══════════════════════════════════════════════════════════
Bayes Theorem: P(y|x) ∝ P(y) · P(x|y)
Naive Assumption: P(x|y) = Π P(xⱼ|y)  (independence)
Decision Rule: ŷ = argmax [log P(y) + Σ log P(xⱼ|y)]

Variants:
  GaussianNB:     P(xⱼ|y) ~ Normal(μⱼc, σ²ⱼc)   [continuous]
  MultinomialNB:  P(xⱼ|y) ∝ count(xⱼ,y) + α      [counts/text]
  BernoulliNB:    P(xⱼ|y) = p^xⱼ(1-p)^(1-xⱼ)    [binary]
  ComplementNB:   Uses complement class            [imbalanced text]

Zero frequency fix: Laplace smoothing (alpha=1.0)

Training: O(n·d), Inference: O(K·d) — very fast!
Online learning: partial_fit() for streaming data

sklearn:
  from sklearn.naive_bayes import MultinomialNB, GaussianNB
  MultinomialNB(alpha=1.0)  — text with TF-IDF/counts
  GaussianNB(var_smoothing=1e-9)  — continuous features
══════════════════════════════════════════════════════════
```
