# Detecção de Fraude em Cartão de Crédito

Notebook que treina e compara modelos de classificação para detectar fraude em transações de
cartão de crédito, num cenário com desbalanceamento severo de classes.

## O problema

A base tem 284.807 transações, das quais apenas ~0,17% são fraude. Nesse cenário, a **acurácia
engana**: um modelo que responde "não é fraude" para tudo já acerta ~99,8% das vezes e, ainda
assim, não detecta nenhuma fraude. Por isso a avaliação aqui é feita por **recall, precisão e
F1 da classe de fraude**, não por acurácia.

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
- Split treino/teste (80/20) com `stratify=Class`, para manter a proporção de fraude nos dois
  conjuntos.

## Modelos comparados

| Modelo | Tratamento do desbalanceamento |
|---|---|
| Regressão Logística (baseline) | `class_weight="balanced"` |
| Random Forest | `class_weight="balanced"` |
| XGBoost | `scale_pos_weight` proporcional ao desbalanceamento |

**Preencher após rodar o notebook** (métricas da classe de fraude, limiar padrão 0.5):

| Modelo | Recall | Precisão | F1 | ROC AUC | PR AUC |
|---|---|---|---|---|---|
| Regressão Logística | | | | | |
| Random Forest | | | | | |
| XGBoost | | | | | |

## Limiar de decisão

O notebook varre limiares de 0.01 a 0.99 e escolhe o que maximiza o F1 da fraude para cada
modelo. **Preencher:** limiar final escolhido, modelo escolhido e a justificativa (ex:
priorização de recall vs. precisão conforme o custo de negócio de deixar fraude passar vs.
gerar falso alarme).

## Explicabilidade (SHAP)

O notebook usa `shap.TreeExplainer` sobre o XGBoost para mostrar quais variáveis mais pesam
na decisão do modelo, com um summary plot geral e a explicação de uma fraude específica.
**Preencher:** quais variáveis o SHAP apontou como mais relevantes para marcar uma transação
como fraude.

## O que mudei em relação ao pipeline da Expert

**Preencher:** liste aqui o que você adaptou/adicionou em relação ao que foi mostrado nas
aulas (ex: oversampling/undersampling testados, GridSearchCV, novas features de
comportamento ao longo do tempo, outro modelo testado).

## Como rodar

1. Abra `fraude_cartao_credito.ipynb` no Google Colab ou Jupyter local;
2. Rode todas as células em sequência — o dataset é baixado automaticamente pela URL;
3. Bibliotecas necessárias: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `shap`,
   `matplotlib`, `seaborn` (`pip install xgboost shap` cobre o que normalmente falta em
   ambientes já com a stack de ciência de dados).

## Ideias para evoluir

- Testar undersampling e oversampling e comparar o recall de cada abordagem;
- Ampliar a busca de hiperparâmetros com `GridSearchCV`;
- Criar variáveis de comportamento ao longo do tempo (transações em sequência);
- Testar outros classificadores e comparar todos pelo recall.
