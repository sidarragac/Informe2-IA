import sys
import os
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Includes.preprocesamiento import preprocesar_dataset
from Includes.generacion_graficas import graficar_resultados

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocesar_dataset()

    model = SVC(kernel='linear', C=1.0, gamma='scale', random_state=42)

    model.fit(X_train, y_train)

    y_val_pred = model.predict(X_val)
    print("Resultados de la validación:")
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    y_test_pred = model.predict(X_test)
    print("Resultados de la prueba:")
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

    # Graficar resultados
    graficar_resultados(y_test, y_test_pred, "svc")

if __name__ == "__main__":
    main()