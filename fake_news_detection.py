import pandas as pd
import numpy as np
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

fake["label"] = 0  # 0 = Fake
true["label"] = 1  # 1 = Real

data = pd.concat([fake, true], axis=0)
data = data[["text", "label"]]  # Only keep 'text' and 'label'
data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def clean_text(text):
    text = text.lower()  # lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # remove punctuation
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]  # remove stopwords
    return " ".join(tokens)

data["clean_text"] = data["text"].apply(clean_text)
X = data["clean_text"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
tfidf = TfidfVectorizer(max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", cm)
def predict_news(news):
    cleaned = clean_text(news)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)
    return "Real News" if pred[0] == 1 else "Fake News"

# Try it
print(predict_news("Government launches new schemes for poor"))
