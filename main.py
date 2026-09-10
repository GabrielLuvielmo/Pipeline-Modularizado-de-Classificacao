import pandas as pd
from data_processing import processar_dados
from model_training import treinar_modelo

# Exemplo de um novo cliente (poucas transações -> tende a dar churn=1)
NOVO_CLIENTE = [
    ('Eletronicos', 120.50),
    ('Eletronicos', 85.00),
    ('Livros', 45.90),
]


def construir_features(transacoes, colunas):
    """Agrega as transações de um cliente no mesmo formato de features
    usado no treino (frequencia, valor_total, valor_<categoria>...)."""
    linha = {c: 0 for c in colunas}
    linha['frequencia'] = len(transacoes)
    linha['valor_total'] = sum(valor for _, valor in transacoes)
    for categoria, valor in transacoes:
        coluna = f'valor_{categoria}'
        if coluna in linha:
            linha[coluna] += valor
    return pd.DataFrame([linha])[colunas]


def main():
    # Etapa 1 - Engenharia de Dados (data_processing.py)
    df_agregado_clean, _ = processar_dados()
    print("Dataset agregado por cliente:")
    print(df_agregado_clean.head())

    # Etapa 2 - Treinamento e avaliação do modelo (model_training.py)
    modelo, metrics = treinar_modelo(df_agregado_clean)
    print(f"\nAcurácia: {metrics['accuracy']:.4f}")
    print("Matriz de Confusão:")
    print(metrics['confusion_matrix'])

    # Etapa 3 - Inferência para um novo cliente de exemplo
    colunas = df_agregado_clean.drop(columns=['churn']).columns
    novo_cliente_df = construir_features(NOVO_CLIENTE, colunas)
    predicao = modelo.predict(novo_cliente_df)[0]
    resultado = 'CHURN' if predicao == 1 else 'NÃO-CHURN'
    print(f"\nPrevisão para o novo cliente: churn = {predicao} -> {resultado}")


if __name__ == '__main__':
    main()
