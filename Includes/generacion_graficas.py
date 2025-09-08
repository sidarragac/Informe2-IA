import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

IMAGES_DIR = "../images/"

def graficar_resultados(y_test, y_pred, method_name):
    # Matriz de confusión
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.title("Matriz de confusión")
    plt.grid(False)
    plt.savefig(f"{IMAGES_DIR}/matriz_confusion_{method_name}.png", dpi=300, bbox_inches="tight")

def grafica_correlacion(df, columnas_numericas):
    corr = df[columnas_numericas].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
    plt.title("Matriz de correlación (variables numéricas)")
    plt.show()

def boxplots_vs_target(df, columnas_numericas):
    for col in columnas_numericas:
        plt.figure(figsize=(6,4))
        sns.boxplot(x="recommended", y=col, data=df)
        plt.title(f"{col} vs Recommended")
        plt.show()