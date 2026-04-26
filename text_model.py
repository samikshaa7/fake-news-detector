import pandas as pd
import pickle
import re

# ---------------- LOAD DATA ----------------
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 1
true["label"] = 0

# 🔥 BALANCE DATASET (VERY IMPORTANT)
fake = fake.sample(n=len(true), random_state=42)

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
    max_features=7000,
    stop_words='english',
    ngram_range=(1,2),
    min_df=2
)

X = vectorizer.fit_transform(X)

# ---------------- SPLIT ----------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL (IMPROVED) ----------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

model = LogisticRegression(max_iter=1000, class_weight='balanced')

model.fit(X_train, y_train)
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, pred)
print("Confusion Matrix:\n", cm)

# ---------------- SAVE ----------------
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(cm, open("cm.pkl", "wb"))
pickle.dump(accuracy, open("accuracy.pkl", "wb"))

print("Model & metrics saved successfully")
