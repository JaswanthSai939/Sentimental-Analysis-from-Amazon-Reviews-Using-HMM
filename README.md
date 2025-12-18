# 📊 Sentiment Analysis from Amazon Reviews Using HMM

## 📌 Overview

This project focuses on **sentiment analysis of Amazon customer reviews** using **probabilistic reasoning and Natural Language Processing (NLP)** techniques. The study applies **Hidden Markov Models (HMM)** and **Naïve Bayes classifiers** to classify customer reviews into **positive, neutral, and negative sentiments**, with a specific focus on the **Grocery and Gourmet Food** domain.

By integrating NLP preprocessing techniques such as **tokenization, stopword removal, lemmatization, and negation handling**, the system effectively captures linguistic patterns and contextual dependencies in textual data. The probabilistic approach enhances sentiment classification accuracy, making the model suitable for **e-commerce feedback analysis and decision-making systems**.

---

## 🎯 Objectives

* To automatically analyze customer sentiment from Amazon product reviews
* To apply **probabilistic models (HMM & Naïve Bayes)** for sentiment classification
* To handle linguistic challenges such as **negations and contextual variations**
* To evaluate model performance using standard classification metrics
* To provide actionable insights for **e-commerce platforms**

---

## ✨ Key Features

* 📝 **Text Preprocessing** (tokenization, lemmatization, stopword removal)
* ❗ **Negation Handling** for improved sentiment interpretation
* 📊 **Probabilistic Sentiment Classification**
* 🔁 **Sequential Modeling using Hidden Markov Models (HMM)**
* 🧠 **Feature Extraction using n-grams and word frequencies**
* 📈 **Performance Evaluation** using accuracy, precision, recall, and F1-score

---

## 🧠 Methodology

### 1️⃣ Data Collection

* Dataset: **Amazon Grocery and Gourmet Food Reviews**
* Total Reviews: **151,254**
* Ratings (1–5) mapped to:

  * Negative
  * Neutral
  * Positive

---

### 2️⃣ Data Preprocessing

* Tokenization
* Stopword removal
* Lemmatization
* Negation handling (e.g., *“not good”*, *“not bad”*)

---

### 3️⃣ Feature Extraction

* N-grams
* Word frequency vectors
* Linguistic features

---

### 4️⃣ Model Training

* **Naïve Bayes Classifier**
* **Hidden Markov Model (HMM)** for sequential dependency learning

---

### 5️⃣ Sentiment Classification

* Classifies reviews into **Positive / Neutral / Negative**
* Probabilistic prediction based on learned features

---

### 6️⃣ Model Evaluation

* Accuracy
* Precision
* Recall
* F1-Score

---

## 📊 Results

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | **80%** |
| Precision | **70%** |
| Recall    | **80%** |
| F1-Score  | **70%** |

### 🔍 Observations

* High recall indicates strong ability to detect relevant sentiments
* Precision can be improved by reducing false positives
* Probabilistic reasoning handles **negation and contextual flow** better than traditional lexicon-based methods

---

## 🛠️ Technologies Used

* **Programming Language:** Python
* **NLP Techniques:**

  * Tokenization
  * Lemmatization
  * Negation Handling
* **Models:**

  * Naïve Bayes
  * Hidden Markov Model (HMM)
* **Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score

---

## 🗂️ Dataset

* **Amazon Grocery and Gourmet Food Reviews Dataset**
* Attributes include:

  * Review text
  * Rating
  * Reviewer ID
  * Product ID (ASIN)
  * Review timestamps

---

## 🌍 Applications

* E-commerce sentiment analysis
* Product feedback evaluation
* Customer satisfaction analysis
* Business intelligence and decision support
* Recommendation system enhancement

---

## 📌 Conclusion

This study demonstrates that **probabilistic reasoning combined with NLP techniques** is an effective approach for sentiment analysis in e-commerce reviews. The model achieves **80% accuracy**, showing strong capability in identifying customer sentiment while handling linguistic complexities such as negation and context.

Although the system performs well, further improvements can be achieved by incorporating **deep learning or hybrid models** to enhance precision and reduce false positives.

---

## 🔮 Future Scope

* Integration with deep learning models (LSTM, RNN)
* Contextual word embeddings (Word2Vec, GloVe, BERT)
* Hybrid probabilistic + deep learning approaches
* Real-time sentiment monitoring dashboards
* Multi-domain sentiment analysis


