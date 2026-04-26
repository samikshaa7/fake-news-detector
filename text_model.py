import pandas as pd
import pickle
import re

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

df = pd.concat([fake, true])
df = df[['text', 'label']]
df = df.dropna()

# ---------------- CLEAN TEXT ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    return text

df['text'] = df['text'].apply(clean_text)

X = df['text']
y = df['label']

# ---------------- TF-IDF ----------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(X)

# ---------------- SPLIT ----------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL COMPARISON ----------------
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000)
}

best_model = None
best_accuracy = 0
best_pred = None

for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)

    acc = accuracy_score(y_test, pred)
    print(name, "Accuracy:", acc)

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = m
        best_pred = pred

print("\nBest Model:", best_model)
print("Best Accuracy:", best_accuracy)

# ---------------- METRICS ----------------
cm = confusion_matrix(y_test, best_pred)
print("Confusion Matrix:\n", cm)

# ---------------- SAVE ----------------
pickle.dump(best_model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(cm, open("cm.pkl", "wb"))
pickle.dump(best_accuracy, open("accuracy.pkl", "wb"))

print("Model & metrics saved successfully")