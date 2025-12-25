import gradio as gr
from transformers import pipeline

# ============================================
# CONFIGURACIÓN DEL MODELO
# ============================================

clasificador = pipeline("image-classification", model="google/vit-base-patch16-224")

# ============================================
# FUNCIÓN PRINCIPAL
# ============================================

def clasificar_imagen(imagen):
    resultados = clasificador(imagen)
    salida = {r["label"]: round(r["score"], 3) for r in resultados}
    return salida

# ============================================
# INTERFAZ GRADIO
# ============================================

iface = gr.Interface(
    fn=clasificar_imagen,
    inputs=gr.Image(type="filepath", label="Subí una imagen"),
    outputs=gr.Label(num_top_classes=5, label="Predicciones del modelo"),
    title="Aplicación de Clasificación de Imágenes",
    description="Desarrollada para Procesamiento Digital de Imágenes y Visión por Computadora"
)

# ============================================
# EJECUCIÓN
# ============================================

if __name__ == "__main__":
    iface.launch(share=True)
