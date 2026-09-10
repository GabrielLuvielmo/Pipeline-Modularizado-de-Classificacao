import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Modelo escolhido: RandomForestClassifier.
# Motivo: não exige normalização das features, captura relações
# não-lineares entre as variáveis e, por ser um ensemble de árvores,
# é mais robusto a overfitting do que uma única árvore de decisão.


def treinar_modelo(df, target_column='churn', test_size=0.3, random_state=42):
    """Recebe os dados tratados, faz o split treino/teste (70/30), treina
    o modelo de classificação e retorna o modelo treinado + métricas
    (acurácia e matriz de confusão)."""
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=200, random_state=random_state)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
    }

    return modelo, metrics


if __name__ == '__main__':
    from data_processing import processar_dados

    df_agregado_clean, _ = processar_dados()
    modelo, metrics = treinar_modelo(df_agregado_clean)

    print(f"Acurácia: {metrics['accuracy']:.4f}")
    print("Matriz de Confusão:")
    print(metrics['confusion_matrix'])
