import sys
import os
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Includes.preprocesamiento import preprocesar_dataset
from Includes.generacion_graficas import graficar_resultados

def main():
    # Cargar datos preprocesados
    X_train, X_val, X_test, y_train, y_val, y_test = preprocesar_dataset()

    # Crear el modelo XGBoost
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Predecir sobre el conjunto de validación
    y_val_pred = model.predict(X_val)
    print("Resultados de la validación:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    # Probar con el conjunto de prueba
    y_test_pred = model.predict(X_test)
    print("Resultados de la prueba:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

    # Graficar resultados
    graficar_resultados(y_test, y_test_pred, "xgboost")

if __name__ == "__main__":
    main()