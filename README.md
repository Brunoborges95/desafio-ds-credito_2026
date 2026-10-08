# Desafio Data Science - Creditas

Modelo que estima a probabilidade de um cliente **pré-aprovado** para o empréstimo com garantia de automóvel ser **enviado para análise de crédito**, para priorizar o atendimento dos leads.

Este repositório tem duas versões do notebook. Este README documenta os erros encontrados na versão original e como cada um foi corrigido na versão 2.

| Arquivo | Conteúdo |
|---|---|
| `Desafio Data Science - Creditas.ipynb` | Versão original, mantida sem alterações para referência |
| `Desafio Data Science - Creditas v2.ipynb` | Versão corrigida, executada de ponta a ponta |
| `dataset.csv`, `description.csv` | Base de dados e dicionário de variáveis |

## Resultado

| | Versão original | Versão 2 |
|---|---|---|
| Linhas de pré-aprovados usadas | 8.870 de 14.999 | 14.998 (4 duplicatas removidas da base) |
| AUC-ROC no teste | 0,782 | **0,825** (IC 95%: 0,809 a 0,841) |
| PR-AUC | não medida | 0,557 (taxa base: 0,218) |
| Lift nos 10% de maior score | não medido | 2,95 |
| Envios capturados nos 20% de maior score | não medido | 50% |
| AUC fora do tempo (20% de leads mais recentes) | não medida | 0,785 |

Os dois valores de AUC não são diretamente comparáveis, porque a versão original foi avaliada em uma amostra reduzida. Na mesma base de teste da versão 2, um LightGBM usando apenas as variáveis da versão original atinge 0,806; as novas variáveis levam a 0,825.

## Erros que afetavam o resultado

### 1. Um terço da base era descartado sem intenção

```python
base_pre2['collateral_debt'] = base_pre2['collateral_debt'].fillna(np.median(base_pre2.collateral_debt))
```

`np.median` devolve `NaN` quando a coluna tem nulos, então o `fillna` não preenchia nada. A variável `Total_value`, calculada a partir de `collateral_debt`, ficava `NaN`, e o filtro de outliers (`x < limiar`) eliminava essas linhas. A base caía de 33.569 para 23.090 linhas.

O descarte era enviesado: entre os pré-aprovados, quem tem `collateral_debt` nulo é enviado para análise em 26,4% dos casos, contra 19,3% dos demais.

**Correção:** nenhuma linha é descartada. O nulo é mantido (o LightGBM trata nativamente) e ganha um indicador próprio, `debt_missing`.

### 2. A métrica de destaque era do alvo errado e medida no treino

A conclusão destacava AUC de 0,98 e acurácia de 95%, que são do modelo de **pré-aprovação**, não do envio para análise. Além disso, esse modelo era avaliado na base inteira, incluindo os 80% usados no treino:

```python
X_train, X_test = X[train_index], X
```

**Correção:** a etapa de pré-aprovação foi removida. Nenhum lead não pré-aprovado é enviado para análise, então a população correta são só os pré-aprovados. Usar a probabilidade de pré-aprovação como variável não é, em si, vazamento do alvo; o problema da versão original era gerá-la dentro da amostra para 80% das linhas, o que não reproduz o que o modelo veria em produção. A versão 2 testa a ideia com a probabilidade gerada fora da amostra e o ganho no modelo final é de 0,0006 de AUC.

### 3. Nomes de variáveis trocados na análise de importância

Na segunda etapa, a matriz de teste recebia os nomes da lista `preditores`, que continha `pre_approved` na posição 9. A matriz, porém, tinha `form_completed` nessa posição e a probabilidade de pré-aprovação na última coluna. Todos os nomes a partir da posição 9 ficavam deslocados.

A frase "a variável pre_approved aumentou a acurácia em 2,89%" se referia, na verdade, a `form_completed`.

**Correção:** os dados permanecem em DataFrames do início ao fim, e a importância por permutação e os valores SHAP usam os nomes das próprias colunas.

### 4. Limiar escolhido na base de teste, maximizando acurácia

O limiar de classificação era escolhido na própria base de teste, o que torna as métricas otimistas. A métrica otimizada era a acurácia, que chegou a 82,2% contra 80,9% de prever sempre "não enviado", com recall de apenas 17,4%.

**Correção:** como o uso é ordenar leads, não há limiar. A avaliação usa AUC, PR-AUC, KS, lift, captura acumulada, Brier, curva de calibração e uma tabela por decil de score.

### 5. Diferença não explicada entre validação e teste

A busca de hiperparâmetros reportava AUC de 0,961 em validação cruzada, mas o teste dava 0,782. O texto mencionava SMOTE, que não aparece em nenhuma célula. Aplicar oversampling antes da validação cruzada é a causa habitual desse tipo de diferença, mas não é possível confirmar pelo notebook.

**Correção:** não há oversampling nem reponderação de classes, que distorcem a probabilidade sem melhorar o ranking. Na versão 2 a validação cruzada (0,815 ± 0,008) e o teste (0,825) são coerentes.

### 6. Separação treino/teste feita depois do pré-processamento

Normalização, codificação de categorias, imputação e remoção de outliers eram ajustadas na base inteira antes da separação.

**Correção:** a separação é o primeiro passo. A análise exploratória e todas as escolhas de variáveis e modelo usam apenas o treino; o teste é usado uma única vez. As variáveis criadas usam só informações da própria linha.

## Outros erros

| Erro na versão original | Correção na versão 2 |
|---|---|
| Filtro de outliers usava `Q1 + const * IQR` em vez de `Q3` e removia linhas, o que não é possível em produção | Sem remoção de linhas. Árvores dependem só da ordem dos valores; a regressão logística de referência usa transformação por quantis ajustada no treino |
| `Total_value` somava a dívida ao valor do carro | `ltv_net` usa o valor do carro menos a dívida; `ltv_gross` usa só o valor do carro |
| `channel`, `landing_page_product`, `utm_term`, `state` e `auto_brand` descartadas. `utm_term` era tratada como identificador, mas é o tipo de dispositivo | Todas incluídas. Canal e landing page estão entre as variáveis mais relevantes |
| `marital_status` descartada apenas pelo excesso de nulos | Descartada e documentada como vazamento: quando preenchida, 92% foram enviados para análise |
| Seção de análise exploratória não comparava nenhuma variável com o alvo | Taxa de envio por categoria, por decil das variáveis numéricas e por coluna nula |
| Correlação de Pearson sobre categorias codificadas com `LabelEncoder` | Removida; substituída pelas taxas de envio por categoria |
| O valor 0,189 do gráfico KS era descrito como "limiar que melhor separa as classes" | KS reportado como métrica |
| Definição de acurácia era cópia da de recall; a calibração era descrita como "distribuir melhor" as probabilidades | Textos reescritos; calibração verificada pela curva de confiabilidade e pelo Brier |
| Conclusão citava AUC de 0,79 (a saída era 0,782) e o texto citava 0,82 do PyCaret, cuja tabela não ficou salva | Todos os números da conclusão conferem com as saídas das células |
| Linhas duplicadas na base não eram verificadas | 4 linhas totalmente duplicadas removidas na carga |
| Sem validação temporal | Treino nos ids mais antigos e teste nos 20% mais recentes |
| Sem modelo de referência | Regressão logística e LightGBM com as variáveis originais como referências |

## Problemas de reprodutibilidade

| Problema na versão original | Correção na versão 2 |
|---|---|
| Células executadas fora de ordem; o notebook não roda de cima para baixo | Notebook executado em sequência, do início ao fim, com semente fixa |
| `rfc` usado na busca de hiperparâmetros antes de ser definido | Corrigido |
| Função `otimizador_xgboost` otimizava um Random Forest, ignorava os argumentos `metrica` e `n_chamadas` e devolvia nomes de parâmetros que não correspondiam ao espaço de busca | Removida. Uma busca bayesiana com Optuna melhorou a AUC em cerca de 0,001, então o modelo usa uma configuração fixa |
| Bloco de PyCaret com alvo `pre_approved` e `sent_to_analysis` entre as variáveis | Removido |
| Caminho absoluto fixo (`C:\Users\bruno\...`) | Caminhos relativos |
| Dependência de `eli5` e `scikit-plot`, que não são mais mantidas | Substituídas por `shap`, `scikit-learn` e `matplotlib` |
| `warnings.filterwarnings("ignore")` global | Filtro restrito a `FutureWarning` e `UserWarning` |

## Limitações da versão 2

- **Mudança ao longo do tempo.** A taxa de envio para análise variou de 12% a 28% ao longo dos ids. Fora do tempo a AUC é 0,785, que é a estimativa mais realista para produção.
- **`form_completed`.** Foi mantida porque aparece preenchida também em leads não pré-aprovados, o que indica registro anterior ao atendimento. Se não estiver disponível no momento da priorização, o modelo sem ela atinge AUC de 0,799.
- **Ganho sobre um modelo linear.** A regressão logística com as mesmas variáveis chega a 0,819. A maior parte do ganho vem das variáveis, não do algoritmo.
- **Alvo.** O envio para análise depende também da capacidade e das decisões do time de atendimento. O ganho real precisa ser medido em um teste A/B.

## Como executar

Requer Python 3 com `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `shap`, `matplotlib`, `seaborn` e `missingno`.

```bash
jupyter nbconvert --to notebook --execute --inplace "Desafio Data Science - Creditas v2.ipynb"
```

O notebook lê `dataset.csv` e `description.csv` da mesma pasta e roda em poucos minutos.
