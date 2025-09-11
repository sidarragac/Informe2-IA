import os
import sys
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Includes.preprocesamiento import preprocesar_dataset
from Includes.generacion_graficas import graficar_resultados

def main():
    # Cargar y preprocesar datos
    X_train, X_val, X_test, y_train, y_val, y_test = preprocesar_dataset()

    # Construir el modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # Probar con el conjunto de validación
    print("Resultados de la validación:")
    y_val_pred = (model.predict(X_val) > 0.5).astype("int").flatten()
    print("Accuracy:", accuracy_score(y_val, y_val_pred))

    # Probar con el conjunto de prueba
    print("Resultados de la prueba:")
    y_test_pred = (model.predict(X_test) > 0.5).astype("int").flatten()
    print("Accuracy:", accuracy_score(y_test, y_test_pred))

    graficar_resultados(y_test, y_test_pred, "neuronal")

if __name__ == "__main__":
    main()