import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocesamiento import preprocesar_dataset

def main():
    X_train, X_val, X_test, y_train, y_val, y_test = preprocesar_dataset()

if __name__ == "__main__":
    main()