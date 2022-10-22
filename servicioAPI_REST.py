'''
Tema: Servicio REST con Flask para atender peticiones de clasificación de imágenes
      con Deep Learning
Autor: Leonardo Martínez González
Referencias:
https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

'''

# Import Flask
from flask import Flask, request, jsonify, redirect
from flask_cors import CORS

# Import Keras
from keras.preprocessing import image

# Import python files
import numpy as np

import requests
import json
import os
from werkzeug.utils import secure_filename
from cargarModelo import cargarModelo

FOLDER_IMAGENES = 'imagenes/cargadas'
EXTENSIONES_PERMITIDAS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print("Puerto: ", port)

# Inicializar el servicio
app = Flask(__name__)
CORS(app)
global loaded_model, graph
loaded_model, graph = cargarModelo()
app.config['FOLDER_IMAGENES'] = FOLDER_IMAGENES

# Función para filtar los archivos permitidos
def archivos_permitidos(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in EXTENSIONES_PERMITIDAS


# Ruta principal
@app.route('/')
def principal():
    return 'Servicio API-REST funcionando, Servidor turismo-jiquilpan ACTIVO'

@app.route('/model/recursos-turisticos-jiquilpan/', methods=['GET', 'POST'])
def default():
    data = {"success": False}
    if request.method == "POST":
        # Verifica que la petición tenga la parte del archivo
        if 'file' not in request.files:
            print('Debes enviar un archivo')
        file = request.files['file']
        # Si el usuario no envia el archivo
        if file.filename == '':
            print('Seleccione un archivo')
        if file and archivos_permitidos(file.filename):
            archivo_imagen = secure_filename(file.filename)
            file.save(os.path.join(app.config['FOLDER_IMAGENES'], archivo_imagen))

            # loading image
            archivo_imagen = FOLDER_IMAGENES + '/' + archivo_imagen
            print("\nArchivo de imagen: ", archivo_imagen)

            imagen_predecir = image.load_img(archivo_imagen, target_size=(224, 224))
            imagen_prueba = image.img_to_array(imagen_predecir)
            imagen_prueba = np.expand_dims(imagen_prueba, axis=0)
            imagen_prueba = imagen_prueba.astype('float32')
            imagen_prueba /= 255

            with graph.as_default():
                result = loaded_model.predict(imagen_prueba)[0]
                # print(result)

                # Resultados
                # prediccion = 1 if (result >= 0.5) else 0
                index = np.argmax(result)
                CLASSES = ['Benito Juárez', 'Fuente de la Aguadora', 'Fuente de los Gallitos', 'Fuente de los pescados', 'Generales Ornelas y Rio Seco', 'Ignacio Zaragoza', 'Lazaro Cárdenas del Río', 'Monumento a Lazaro Cárdenas', 'Lucia de la Paz']
                ClassPred = CLASSES[index]
                ClassProb = result[index]

                print("Index:", index)
                print("Predicción:", ClassPred)
                print("Prob: {:.2%}".format(ClassProb))

                # Insertar los resultados al JSON
                data["predictions"] = []
                r = {"label": ClassPred, "score": float(ClassProb)}
                data["predictions"].append(r)

                # Respuesta satisfactoria
                data["success"] = True

    return jsonify(data)


# Ejecutar la aplicación
app.run(host='0.0.0.0', port=port, threaded=False)