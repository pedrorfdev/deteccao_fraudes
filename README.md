# Detecção de Fraude em Cartão de Crédito

Notebook que treina e compara modelos de classificação para detectar fraude em transações de
cartão de crédito, num cenário com desbalanceamento severo de classes.

## O problema

A base tem 284.807 transações, das quais apenas 492 (~0,17%) são fraude. Nesse cenário, a
**acurácia engana**: um modelo que responde "não é fraude" para tudo já acerta ~99,8% das vezes
e, ainda assim, não detecta nenhuma fraude. Por isso a avaliação aqui é feita por **recall,
precisão e F1 da classe de fraude**, não por acurácia.

## Dataset

Carregado diretamente da URL abaixo dentro do notebook — não fica versionado no repositório:

```
https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv
```

Colunas: `Time`, `Amount`, `Class` (0 = normal, 1 = fraude) e `V1`...`V28` (variáveis
anonimizadas por PCA).

## Preparação dos dados

- Criação de `LogAmount` (log do valor de transação, já que `Amount` é muito assimétrico);
- Padronização de `Time`, `Amount` e `LogAmount` com `StandardScaler`;
- Split treino/teste (80/20) com `stratify=Class`, mantendo a proporção de fraude nos dois
  conjuntos (~0,17% em treino e em teste).

## Modelos comparados

| Modelo | Tratamento do desbalanceamento |
|---|---|
| Regressão Logística (baseline) | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |

Métricas da classe de fraude, limiar padrão 0.5:

| Modelo | Recall | Precisão | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Regressão Logística | 0.9184 | 0.0614 | 0.1151 | 0.9719 | 0.7164 |
| Random Forest | 0.7551 | 0.9610 | 0.8457 | 0.9569 | 0.8653 |

A Regressão Logística detecta quase toda fraude, mas gera muito falso alarme (precisão baixa).
O Random Forest é bem mais equilibrado. Vale notar que o ROC AUC engana um pouco aqui: como a
classe normal é enorme (56.864 casos no teste), a taxa de falso positivo fica achatada e infla
o AUC mesmo quando o modelo erra bastante em termos absolutos. O **PR AUC é a métrica mais
confiável nesse desbalanceamento**, e nele o Random Forest é melhor (0.8653 vs 0.7164).

## Limiar de decisão

O notebook varre limiares de 0.01 a 0.99 e escolhe, para cada modelo, o que maximiza o F1 da
classe de fraude:

| Modelo | Limiar ótimo | Recall | Precisão | F1 |
|---|---|---|---|---|
| Regressão Logística | 0.99 | 0.8469 | 0.5764 | 0.6860 |
| Random Forest | 0.27 | 0.8367 | 0.9425 | 0.8865 |

**Modelo e limiar final escolhidos: Random Forest, limiar 0.27.** Com esse ajuste, o recall
sobe de 75,5% (limiar padrão) para 83,7%, sem sacrificar muito a precisão (cai de 96,1% para
94,3%). É o melhor equilíbrio entre detectar fraude e não gerar excesso de falso alarme — e
o F1 de 0.8865 é o melhor resultado entre todas as combinações testadas.

## Explicabilidade

Importância de variáveis do Random Forest (`feature_importances_`), mostrando quais colunas
mais pesam nas decisões do modelo.

**Preencher:** quais variáveis apareceram como mais relevantes para marcar uma transação como
fraude.

## O que mudei em relação ao pipeline da Expert

- Comparei apenas 2 modelos (Regressão Logística e Random Forest) em vez de incluir XGBoost;
- Usei importância de variáveis do Random Forest em vez de SHAP para a explicabilidade;
- Ajustei o limiar de decisão maximizando o F1 da fraude para cada modelo, em vez de usar
  só o limiar padrão de 0.5.

## Como rodar

1. Abra `fraude_cartao_credito_simples.ipynb` no Google Colab ou Jupyter local;
2. Rode todas as células em sequência — o dataset é baixado automaticamente pela URL;
3. Bibliotecas necessárias: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
   (já vêm por padrão no Google Colab, não precisa instalar nada extra).

## Ideias para evoluir

- Testar undersampling e oversampling e comparar o recall de cada abordagem;
- Ampliar a busca de hiperparâmetros com `GridSearchCV`;
- Testar outros modelos (ex: XGBoost, LightGBM) e comparar todos pelo recall;
- Criar variáveis de comportamento ao longo do tempo (transações em sequência).
