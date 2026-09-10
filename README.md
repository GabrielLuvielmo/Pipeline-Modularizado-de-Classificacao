# Pipeline Modularizado de Classificação de Churn

Atividade Prática — Unidade Curricular de Engenharia de Dados & MLOps (Trabalho em Trio).

## Cenário de Negócio

Uma loja online deseja identificar clientes inativos com risco de não voltarem a comprar (**Churn**). Este projeto processa um dataset bruto de transações, agrega os dados por cliente, treina um modelo de classificação e realiza a previsão de churn para um novo cliente.

## Estrutura do Projeto

```
projeto_modular_trio/
├── data/
│   └── raw_transactions.csv   # Dataset bruto de transações
├── data_processing.py          # Engenharia de Dados (Pessoa A)
├── model_training.py           # Treino e avaliação do modelo (Pessoa B)
└── main.py                     # Orquestração e inferência (Pessoa C)
```

## Módulos

### `data_processing.py`
Lê `data/raw_transactions.csv` (colunas: `id_transacao`, `id_cliente`, `categoria`, `valor`, `data`) e aplica Tabelas Dinâmicas (`pd.pivot_table`) para agregar, por cliente:
- `frequencia`: número de transações;
- `valor_total`: soma de todos os valores gasto;
- `valor_<categoria>`: soma do valor gasto em cada categoria.

Gera a coluna `churn` (**1** se `frequencia < 8`, senão **0**) e trata valores nulos (`fillna(0)`) para clientes sem compras em determinada categoria.

Função principal: `processar_dados(path)` → retorna `(df_agregado_clean, df_final)`.

### `model_training.py`
Recebe o dataset agregado por cliente (`df_agregado_clean`), realiza o split treino/teste (**70/30**, estratificado pela classe) e treina um `RandomForestClassifier`.

**Por que RandomForest?** Não exige normalização das features, captura relações não-lineares entre as variáveis e, por ser um ensemble de árvores, é mais robusto a overfitting do que uma única árvore de decisão.

Função principal: `treinar_modelo(df)` → retorna `(modelo, metrics)`, com `metrics` contendo `accuracy` e `confusion_matrix`.

### `main.py`
Importa os módulos acima e orquestra o pipeline completo:
1. Processa os dados brutos.
2. Treina e avalia o modelo (imprime acurácia e matriz de confusão).
3. Constrói as features de um novo cliente de exemplo e prevê seu churn.

## Como Executar

Instale as dependências:

```bash
pip install pandas scikit-learn
```

Execute o pipeline completo:

```bash
python main.py
```

## Exemplo de Saída

```
Dataset agregado por cliente:
            frequencia  valor_total  valor_Alimentos  valor_Casa  valor_Eletronicos  valor_Livros  valor_Roupas  churn
id_cliente
1                   14      1678.29           514.98      448.08               0.00        171.67        543.56      0
2                    5       878.52           259.85      252.72               0.00         79.24        286.71      1
3                   19      3422.58           183.22     1219.70             461.09        264.48       1294.09      0
4                    2       410.44            51.79        0.00             358.65          0.00          0.00      1
5                   15      2810.58           613.29      519.09            1284.53        110.76        282.91      0

Acurácia: 1.0000
Matriz de Confusão:
[[8 0]
 [0 7]]

Previsão para o novo cliente: churn = 1 -> CHURN
```

## Nota Importante: Acurácia de 100%

A coluna `churn` é derivada diretamente da `frequencia` (`churn = 1 se frequencia < 8`), e essa mesma coluna permanece como **feature de entrada** do modelo. Isso faz o RandomForest essencialmente reaprender esse limiar, o que caracteriza um vazamento de dados (**data leakage**) e explica a acurácia de 100%.

Isso é esperado neste exercício didático, já que segue a especificação da atividade — mas em um cenário real de produção seria necessário remover `frequencia` das features (mantendo-a apenas para gerar o `churn`), para obter uma avaliação mais realista do poder preditivo do modelo com base apenas nos gastos por categoria.

## Divisão de Tarefas (Trio)

| Responsável | Módulo | Responsabilidade |
|---|---|---|
| Pessoa A | `data_processing.py` | Engenharia de Dados: leitura, `pivot_table` e geração da flag `churn` |
| Pessoa B | `model_training.py` | Ciência de Dados: split treino/teste, treino do modelo e métricas |
| Pessoa C | `main.py` | MLOps & Integração: orquestração do pipeline e inferência final |
