'''
Tema: Deep Learning aplicado al reconocimiento de recursos turísticos
      de la ciudad de Jiquilpan Michpacán Pueblo Mpagico
Autor: Leonardo Martínez González
Fundamento:

'''

import tensorflow as tf
from keras.models import load_model

# Función para cargar el modelo del archivo: recursos_turisticos_model_full_tf2.h5
def cargarModelo():

    ARCHIVO_MODELO = "recursos_turisticos_model_full_tf2.h5"
    RUTA_MODELO = "model/tf2x/keras/full"

    # Cargar la RNA desde disco
    modelo_cargado = load_model(RUTA_MODELO + "/" + ARCHIVO_MODELO)
    print("Modelo cargado de disco . . . ", modelo_cargado)

    graph = tf.get_default_graph()
    return modelo_cargado, graph