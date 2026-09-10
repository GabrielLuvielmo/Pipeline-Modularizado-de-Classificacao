import pandas as pd


def processar_dados(path='data/raw_transactions (1).csv'):
    """Lê o CSV bruto de transações e retorna (df_agregado_clean, df_final)."""
    df = pd.read_csv(path, sep=',', encoding='utf-8')

    # 1. Agregação de frequência (contagem de transações)
    df_frequencia = pd.pivot_table(
        df,
        values='id_transacao',
        index='id_cliente',
        aggfunc='count'
    )
    df_frequencia.columns = ['frequencia']

    # 2. Agregação de valor total por cliente
    df_valor_total = pd.pivot_table(
        df,
        values='valor',
        index='id_cliente',
        aggfunc='sum'
    )
    df_valor_total.columns = ['valor_total']

    # 3. Agregação de valor por categoria
    df_valor_categoria = pd.pivot_table(
        df,
        values='valor',
        index='id_cliente',
        columns='categoria',
        aggfunc='sum'
    )
    df_valor_categoria.columns = [f'valor_{cat}' for cat in df_valor_categoria.columns]

    # 4. Combinar as agregações
    df_agregado = pd.concat([df_frequencia, df_valor_total, df_valor_categoria], axis=1)

    # 5. Gerar coluna churn: 1 se frequência < 8, senão 0
    df_agregado['churn'] = (df_agregado['frequencia'] < 8).astype(int)

    # 6. LIMPAR NaN ANTES DE USAR - AQUI! 
    df_agregado_clean = df_agregado.fillna(0)

    # 7. Fazer merge com o dataframe original usando df_agregado_clean
    df_final = df.merge(df_agregado_clean, left_on='id_cliente', right_index=True, how='left')

    return df_agregado_clean, df_final


if __name__ == '__main__':
    df_agregado_clean, df_final = processar_dados()

    # Visualizar resultados
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    print(df_agregado_clean.head(10))
    print("\nDataframe final com todas as informações:")
    print(df_final.head(10))
