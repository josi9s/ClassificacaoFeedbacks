import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

rodando = True
exemplos = "../data/examples.json"

def lerJson(exemplos):
  with open(exemplos, 'r') as arquivo:
    dados = json.load(arquivo)

  feedbacks = [feedbacks['feedback'] for feedbacks in dados]
  labels = [label['label'] for label in dados]
  return feedbacks,labels


feedbacks,labels = lerJson(exemplos)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(feedbacks)

X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

model = MultinomialNB()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

precisao = accuracy_score(y_test, y_pred)
print(f"Acurácia do modelo: {precisao * 100:.2f}%")

while(rodando):
    novoFeedback = input("Insira um novo feedback: ")
    X_novos = vectorizer.transform([novoFeedback])
    predicao = model.predict(X_novos)
    print(predicao[0])
