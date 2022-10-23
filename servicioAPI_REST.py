'''
Tema: Servicio REST con Flask para atender peticiones de clasificación de imágenes
      con Deep Learning
Autor: Leonardo Martínez González
Referencias:
https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037

'''

#Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

#Tensorflow image
from tensorflow.keras.preprocessing import image

#Algunas librerias
import numpy as np
from werkzeug.utils import secure_filename

#Importar la fucnión para cargar el modelo
from cargarModelo import cargarModeloH5

#Args
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--port", required=True, help="Service PORT number is required.")
args = vars(ap.parse_args())

#Configurando el puerto del servicio
port = args['port']
print("Puerto: ", port)

#Parametros
UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Initializar la App de Flask
app = Flask(__name__)
CORS(app)

global loaded_model
loaded_model = cargarModeloH5()

def allowed_file(archivo):
    return '.' in archivo and archivo.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Definiendo la ruta por default
@app.route('/')
def main_page():
	return 'Servicio API-REST funcionando, App de Tursimo a la espera de imágenes'

# Ruta para realizar la predicción
@app.route('/model/predict/',methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        # Verificar si existe la parte del archivo
        if 'file' not in request.files:
            print('Envie un archivo en su petición')
        file = request.files['file']
        # Verificar si se envió el archivo
        if file.filename == '':
            print('No seleccionó el archivo de imagen')
        if file and allowed_file(file.filename):
            print("\nArchivo recibido: ",file.filename)
            filename = secure_filename(file.filename)
            tmpfile = ''.join([UPLOAD_FOLDER ,'/',filename])
            file.save(tmpfile)
            print("\nFilename stored:",tmpfile)

            #loading image
            image_to_predict = image.load_img(tmpfile, target_size=(224, 224))
            test_image = image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255.0

            predictions = loaded_model.predict(test_image)[0]
            index = np.argmax(predictions)
            CLASSES = ['Benito Juárez', 'Fuente de la Aguadora', 'Fuente de los Gallitos', 'Fuente de los pescados', 'Generales Ornelas y Rio Seco', 'Ignacio Zaragoza', 'Lazaro Cárdenas del Río', 'Monumento a Lazaro Cárdenas', 'Lucia de la Paz']
            ClassPred = CLASSES[index]
            ClassProb = predictions[index]

            print("Clases: ", CLASSES)
            print("Predicción: ",predictions)
            print("Predicción Index: ", index)
            print("Predicción Label: ", ClassPred)
            print("Predicción Prob: {:.2%} ".format(ClassProb))

            #Agregamos los resultados al JSON data
            data["predictions"] = []
            r = {"label": ClassPred, "score": float(ClassProb)}
            data["predictions"].append(r)

            #Todo bien
            data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)