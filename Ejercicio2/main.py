import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Includes.preprocesamiento import preprocesar_dataset

def main():
    # 1. Cargar y preprocesar datos
    X_train, X_val, X_test, y_train, y_val, y_test = preprocesar_dataset()

    # 2. Construir el modelo
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # 3. Compilar el modelo
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    # 4. Entrenar el modelo
    model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_val, y_val), verbose=1)
    
    # 5. Evaluar el modelo
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'\nPrecisión en los datos de prueba: {test_acc:.4f}')

    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int)

    # Generar y guardar la matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Matriz de confusión - Red Neuronal")
    plt.grid(False)


    # Guardar la imagen en la carpeta "images"
    plt.savefig("../images/matriz_confusion_neuronal.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()