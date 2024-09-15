# Algoritmo de Classificação de Feedbacks com Naive Bayes

Este algoritmo utiliza um classificador Naive Bayes multinomial para categorizar feedbacks com base em exemplos fornecidos em um arquivo JSON. O objetivo é rotular novos feedbacks inseridos pelo usuário como "negativo ou "positivo".

## Implementação de Bibliotecas

Ferramentas utilizadas no script:

- Biblioteca "json" do Python para armazenar os dados em um arquivo JSON

- Funções importadas da Biblioteca scikit-learn/sklearn, uma biblioteca de aprendizado de máquina em Python que fornece ferramentas simples e eficientes para análise de dados e modelagem preditiva:

   **CountVectorizer**: 
     - Para transformar o texto em uma matriz de contagens de palavras.
  
   **MultinomialNB**: 
     - Implementa o classificador Naive Bayes para dados multinomiais.
  
   **train_test_split**: 
     - Divide os dados em partes de treino e teste.
  
   **accuracy_score**: 
     - Para medir a acurácia do modelo.

## Explicação do Código

O código começa declarando uma variável **_exemplos_**, que guarda o caminho para o arquivo JSON com os exemplos de feedbacks a serem usados no treinamento:
```
exemplos = "../data/examples.json"
```

O arquivo JSON utilizado para treinar o modelo de classificação possui a seguinte estrutura:

```
[
  {
    "feedback": "Ótimo atendimento e equipe amigável!",
    "label": "positivo"
  }
]
```
**_feedback_**: Contém o texto do feedback fornecido. Este é o dado de entrada que será analisado pelo modelo de aprendizado de máquina. Exemplos incluem frases como "Ótimo atendimento e equipe amigável!".

**_label_**: Contém a categoria associada ao feedback. Este é o dado de saída que o modelo deve prever. Os rótulos possíveis incluem "positivo" e "negativo".

Logo em seguida, é criada uma função que lê o arquivo JSON:

```
def lerJson(exemplos):
  with open(exemplos, 'r') as arquivo:
    dados = json.load(arquivo)

  feedbacks = [feedback['feedback'] for feedback in dados]
  labels = [label['label'] for label in dados]
  return feedbacks,labels
```

A função **_lerJson_** abrirá um arquivo JSON, carregará seus dados, e extrairá duas listas: uma contendo os textos dos feedbacks e outra contendo seus rótulos associados.

Depois as listas serão desempacotadas em duas variáveis, **_feedbacks_** e **_labels_**, para que possam ser usadas separadamente no código:

```
feedbacks,labels = lerJson(exemplos)
```
Depois será criada uma instância do **_CountVectorizer_**, que é uma classe capaz de converter texto em uma matriz de contagem de palavras. Cada documento (feedback) é transformado em um vetor numérico onde cada valor representa a contagem de uma palavra específica, fazer isso é necessário porquê o modelo de ML só trabalha com valores númericos.

```
vectorizer = CountVectorizer()
```

Feito isso, será utilizado um método chamado **_fit_transform_** nessa instância, que receberá como parâmetro a variável **_feedbacks_**. Essa função basicamente vai criar uma matriz esparsa, em que cada linha representará um feedback e cada coluna representará uma palavra do vocabulário, e os valores vão indicar quantas vezes cada palavra aparece em cada feedback. No final disso, essa matriz vai ser armazenada na variável **_X_**.

```
X = vectorizer.fit_transform(feedbacks)
```
