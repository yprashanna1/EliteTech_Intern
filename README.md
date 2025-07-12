# 💼 EliteTech Internship – Machine Learning Projects

Welcome to my submission for the **EliteTech Internship** under the **Machine Learning track**.  
This repository contains solutions to all 4 assigned tasks, completed using **Python**, **Google Colab**, and popular ML libraries like `scikit-learn`, `TensorFlow`, and `Surprise`.

Each task is provided as a **clean, commented Jupyter notebook** with outputs and visualizations.  
---

## 📌 Table of Contents

1. [Task 1 – Decision Tree Classification](#task-1--decision-tree-classification)
2. [Task 2 – Sentiment Analysis with NLP](#task-2--sentiment-analysis-with-nlp)
3. [Task 3 – Image Classification using CNN](#task-3--image-classification-using-cnn)
4. [Task 4 – Recommendation System](#task-4--recommendation-system)

---

## 🎯 Task 1 – Decision Tree Classification

> Build and visualize a Decision Tree using Scikit-Learn.

- 📁 **File:** `decision_tree.ipynb`
- 📊 Dataset: Iris Dataset (built-in from `sklearn.datasets`)
- 🔧 Tools: `DecisionTreeClassifier`, `train_test_split`, `plot_tree`

### ✅ Features:
- Visualized decision tree using `sklearn.tree.plot_tree()`
- Clean and minimal code with comments
- Achieved high accuracy on a simple flower classification task

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
plot_tree(model, filled=True)
```

---

## 🧠 Task 2 – Sentiment Analysis with NLP

> Perform sentiment analysis using TF-IDF and Logistic Regression.

- 📁 **File:** `sentiment_analysis.ipynb`
- 🗃️ Dataset: Sample product reviews (positive/negative)
- 🛠️ Tools: `TfidfVectorizer`, `LogisticRegression`, `sklearn.metrics`

### ✅ Features:
- Text preprocessing and vectorization
- TF-IDF feature extraction
- Sentiment classification model
- Evaluation using accuracy, precision, recall, and F1-score

```python
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

## 🖼️ Task 3 – Image Classification using CNN

> Build a Convolutional Neural Network to classify images.

- 📁 **File:** `cnn_image_classification.ipynb`
- 🗃️ Dataset: Fashion MNIST (70,000 grayscale clothing images)
- 🔧 Tools: `TensorFlow`, `Keras`, `Conv2D`, `MaxPooling2D`

### ✅ Features:
- CNN model trained and evaluated on Fashion MNIST
- Achieved over 85% test accuracy
- Includes confusion matrix and sample predictions

### 📦 Model Exported:
```python
model.save('fashion_mnist_cnn.h5')
```
🔗 [Download Saved Model from Google Drive](https://drive.google.com/your-cnn-model-link)

---

## 🎬 Task 4 – Recommendation System

> Build a movie recommendation system using matrix factorization (SVD).

- 📁 **File:** `recommendation_system.ipynb`
- 📊 Dataset: MovieLens 100k
- 📦 Library: `Surprise`

### ✅ Features:
- Used `SVD` from `surprise` to model user preferences
- Evaluated using RMSE and MAE
- Generated Top-N movie recommendations for a sample user
- Visualized rating distribution

```python
from surprise import SVD, Dataset
model = SVD()
model.fit(trainset)
predictions = model.test(testset)
```

---

## 🧾 Final Notes

- All notebooks are optimized for **Google Colab** (no setup needed)
- Each notebook includes comments, visualizations, and results
- Completed as part of the **EliteTech Internship Program 2025**

Feel free to explore, run, and extend the work in this repo!  
Happy Learning! 🚀

---
