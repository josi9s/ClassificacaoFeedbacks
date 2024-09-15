# Algoritmo de Classificação de Feedbacks com Naive Bayes

Este algoritmo utiliza um classificador Naive Bayes multinomial para categorizar feedbacks com base em exemplos fornecidos em um arquivo JSON. O objetivo é rotular novos feedbacks inseridos pelo usuário como "negativo ou "positivo".

## Implementação de Bibliotecas

Ferramentas utilizadas no script:

- Biblioteca *json* do Python para armazenar os dados em um arquivo JSON

- Funções importadas da Biblioteca *scikit-learn/sklearn*, uma biblioteca de aprendizado de máquina em Python que fornece ferramentas simples e eficientes para análise de dados e modelagem preditiva:

   **CountVectorizer**: 
     - Para transformar o texto em uma matriz de contagens de palavras.
  
   **MultinomialNB**: 
     - Implementa o classificador Naive Bayes para dados multinomiais.
  
   **train_test_split**: 
     - Divide os dados em partes de treino e teste.
  
   **accuracy_score**: 
     - Para medir a acurácia do modelo.

## Explicação do Código

Eu comecei o código declarando uma variável **_exemplos_**, que guarda o caminho para o arquivo JSON com os exemplos de feedbacks a serem usados no treinamento:
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
Depois eu criei uma instância do **_CountVectorizer_**, que é uma classe capaz de converter o texto em uma matriz de contagem de palavras. Cada documento (feedback) é transformado em um vetor numérico onde cada valor representa a contagem de uma palavra específica, fazer isso é necessário porquê o modelo de ML só trabalha com valores númericos.

```
vectorizer = CountVectorizer()
```

Feito isso, eu utilizei um método chamado **_fit_transform_** nessa instância, que recebe como parâmetro a variável **_feedbacks_**. Essa função basicamente cria uma matriz esparsa, em que cada linha representa um feedback e cada coluna representa uma palavra do vocabulário, e os valores vão indicar quantas vezes cada palavra aparece em cada feedback. No final disso, essa matriz é armazenada na variável **_X_**.

```
X = vectorizer.fit_transform(feedbacks)
```

Logo depois eu utilizei um método chamado **_train_test_split()_**, que é responsável por fazer a divisão dos dados em dois grupos: **treinamento** e **teste**.

```
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
```

Essa função recebe os seguintes parâmetros:

- **_X_** : São os vetores numéricos gerados pelo **_fit_transform_**
- **_labels_** : São os rótulos ligados a cada um dos feedbacks.
- **_test_size_** : Delimitador que indica a porcentagem dos dados que serão utilizados para teste, o restante será usado para treino.
- **_random_state_** : Garante que a divisão dos dados entre treino e teste seja sempre a mesma a cada execução do código.   

    _NOTA*: O valor "42" do **random_state** é arbitrário. O **random_state** não melhora diretamente a acurácia, mas durante meus testes, aparentou ser um bom valor, o que acabou otimizando a performance do modelo._

Essa linha retorna quatro variáveis:

- **_X_train_** : Os vetores de feedback que serão usados para treinar o modelo.
- **_X_test_** : Os vetores de feedback que serão usados para testar o modelo.
- **_y_train_** : Os rótulos correspondentes aos feedbacks de treinamento
- **_y_train_** : Os rótulos correspondentes aos feedbacks de teste.


Em seguida, criei uma instância chamada **_model_** do Modelo Naive Bayes Multinomial, que é apropriada para lidar com tipos de dados textuais.
```
model = MultinomialNB()
```

Depois usei o método **_fit()_** para treinar o modelo. Essa função recebe como parâmetro os vetores e rótulos de treino extraídos da função **_train_test_split()_**
```
model.fit(X_train, y_train)
```

Aqui usei a função **_predict()_** no **_model_**, que é responsável por fazer o modelo tentar prever o rótulo dos dados fornecidos. Ela recebe como parâmetro a variável **_X_test_**. A previsão resultante será armazenada na variável **_y_pred_**.

```
y_pred = model.predict(X_test)
```

Nessa parte eu utilizei o método **_accuracy_score()_** para medir a acurácia do modelo. A função compara os rótulos reais dos dados de teste com os rótulos previstos pelo modelo (y_pred). A acurácia indica a porcentagem de previsões corretas feitas pelo modelo.

```
precisao = accuracy_score(y_test, y_pred)
```
Em seguida, o valor da acurácia é multiplicado por 100 para ser exibido em formato percentual, e é impresso no console com duas casas decimais.
```
print(f"Acurácia do modelo: {precisao * 100:.2f}%")
```

Após isso, criei uma variável **_rodando_** que vai receber o valor *True* que será recebida dentro de um loop *while*. Ela serve como parâmetro para controlar o loop *while*, permitindo que o programa continue pedindo novos feedbacks até que ele seja explicitamente interrompido.

```
rodando = True
```

O loop while vai permanecer ativo enquanto **_rodando_** for *True*. Dentro do loop, o programa solicita ao usuário que insira um novo feedback usando a função **_input()_** e guarda a resposta em uma varíavel chamada **_novoFeedback_**.
```
while(rodando):
    novoFeedback = input("Insira um novo feedback: ")
```

O vectorizer já foi treinado anteriormente com os feedbacks de treino, então aqui ele é utilizado para transformar o **_novoFeedback_** em uma matriz de contagem de palavras, assim como foi feito com os feedbacks de treino. Essa função será armazenada numa varíavel chamada **_X_novos_**.

```
    X_novos = vectorizer.transform([novoFeedback])
```
O modelo usa a função **_predict()_** para fazer a previsão da categoria do novo feedback, usando como parâmetro a variável **_X_novos_**. Essa função será armazenada numa varíavel chamada **_predicao_**.
  
```
    predicao = model.predict(X_novos)
```

Por fim, o resultado de **_predicao_** é exibido no terminal. O valor *predicao[0]* contém o rótulo previsto pelo modelo.

```
    print(predicao[0])
```
