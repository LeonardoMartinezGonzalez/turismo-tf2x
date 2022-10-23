'''
Tema: Deep Learning aplicado al reconocimiento de recursos turísticos
      de la ciudad de Jiquilpan Michpacán Pueblo Mpagico
Autor: Leonardo Martínez González
Fundamento:

'''
import tensorflow as tf

def cargarModeloH5():
    ARCHIVO_MODELO = "recursos_turisticos_model_full_tf2.h5"
    RUTA_MODELO = "../model/"

    # Cargar el modelo desde disco
    loaded_model = tf.keras.models.load_model(RUTA_MODELO + ARCHIVO_MODELO)
    print(ARCHIVO_MODELO, " Cargando . . . ", loaded_model)

    return loaded_model

