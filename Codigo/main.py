from main_ui import *
from os import listdir
import numpy as np
from os.path import expanduser, isfile, join
from PyQt5.QtWidgets import QFileDialog
import glob
import ctypes
from nltk import *
from collections import Counter
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pandas as pd
from joblib import dump, load
import pickle




class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    dirModelo = None
    dirDespoblacion = "..\\Datos\\despoblacion"
    dirNoDespoblacion = "..\\Datos\\no_despoblacion"
    dirSinMarcar = "..\\Datos\\unlabeled"
    dirRutaNoticias = None
    noticiasTesting = None
    listaPalabrasContadasDespoblacion = []
    listaPalabrasContadasNoDespoblacion = []
    listaPalabrasContadasSinMarcar = []
    listaPalabras = []

    

    #Inicializamos el stopwords
    stop_words = set(corpus.stopwords.words('spanish'))

    #Inicializamos el stemmer
    stemmer = stem.SnowballStemmer('spanish')



    def __init__(self, *args, **kwargs):
        #Inicializacion de la ventana y listeners
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.btnRutaDespoblacion.clicked.connect(lambda: self.abrirRuta(0))
        self.btnRutaNoDespoblacion.clicked.connect(lambda: self.abrirRuta(1))
        self.btnRutaSinMarcar.clicked.connect(lambda: self.abrirRuta(2))
        self.btnRutaNoticias.clicked.connect(lambda: self.abrirRuta(3))
        self.btnRutaModelo.clicked.connect(self.abrirModelo)
        self.btnGenerarModelo.clicked.connect(self.generarModelo)
        
    
    def abrirRuta(self, i):
        #Guardamos en la variable la ruta seleccionada a traves de la ventana.
        my_dir = str(QFileDialog.getExistingDirectory(self, "Abre una carpeta", expanduser("~"), QFileDialog.ShowDirsOnly))

        #Dependiendo del boton presionado se guardara la ruta en diferentes variables
        if i == 0:
            self.lblRutaDespoblacion.setText(my_dir)
            self.dirDespoblacion = my_dir

        elif i == 1: 
            self.lblRutaNoDespoblacion.setText(my_dir)
            self.dirNoDespoblacion = my_dir

        elif i == 2: 
            self.lblRutaSinMarcar.setText(my_dir)
            self.dirSinMarcar = my_dir

        elif i == 3: 
            self.lblRutaNoticias.setText(my_dir)
            self.dirRutaNoticias = my_dir

    def abrirModelo(self):
        modelo = QFileDialog.getOpenFileName(self, 'Abrir modelo', 'c:\\', "Modelo (*.*)")
        self.lblRutaModelo.setText(modelo[0])
        self.dirModelo = modelo[0]
        modelo = self.cargarModelo(self.dirModelo)
    

    def getFicherosDirectorio(self, dir):
        #Crea una lista con la ruta de todos los ficheros
        ficheros = glob.glob(dir + "/*.txt")
        return ficheros


    def generarModelo(self):
        #Comprobacion de que todas las rutas se han seleccionado
        if self.dirDespoblacion == None or self.dirNoDespoblacion == None or self.dirSinMarcar == None:
            #Mensaje de error
            ctypes.windll.user32.MessageBoxW(0, "Debes elegir una ruta para todos las noticias", "Error al recuperar noticias", 0)
        else:
            #Obtenemos las rutas de los archivos
            noticiasNoDespoblacion = self.getFicherosDirectorio(self.dirNoDespoblacion)
            noticiasDespoblacion = self.getFicherosDirectorio(self.dirDespoblacion)
            noticiasSinMarcar = self.getFicherosDirectorio(self.dirSinMarcar)

            #Procesamiento de texto
            listaPalabrasContadasDespoblacion = self.getPalabrasContadas(noticiasDespoblacion, 1)
            listaPalabrasContadasNoDespoblacion = self.getPalabrasContadas(noticiasNoDespoblacion, 0)

            print("Lista Palabras Contadas Despoblacion")
            print(listaPalabrasContadasDespoblacion)
            print("Lista Palabras Contadas No Despoblacion")
            print(listaPalabrasContadasNoDespoblacion)

            listaPalabrasContadas = listaPalabrasContadasDespoblacion + listaPalabrasContadasNoDespoblacion

            print("Lista Palabras Contadas")
            print(listaPalabrasContadas)


            #Obtenemos el counter total para guardar todas las palabras en un array
            counterTotal = Counter()
            for counter in listaPalabrasContadas:
                counterTotal += counter
            
            print("Counter Total:")
            print(counterTotal)


            listaPalabras = []
            for palabra in counterTotal:
                listaPalabras.append(palabra)

            print("listaPalabras")
            print(listaPalabras)            

            #Creamos el dataframe que le pasaremos al modelo con todas las palabras en las columnas y una fila por noticia 
            df = pd.DataFrame(listaPalabrasContadas, columns=listaPalabras).fillna(0)
            print(df)

            #En X tenemos todas las palabras y su cuenta, es decir los datos que vamos a necesitar para predecir
            X = df.drop("esDespoblacion", axis=1)

            #En y tenemos la columna que queremos que prediga
            y = df["esDespoblacion"]


            #Divide los datos en training y testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training y 20% test

            #Creamos el clasificador y lo entrenamos
            clf = DecisionTreeClassifier()    
            clf = clf.fit(X_train,y_train)

            #Guardamos en una variable el resultado de la prediccion
            y_pred = clf.predict(X_test)

            #Imprimimos el % de acierto en consola
            print(metrics.accuracy_score(y_test, y_pred))
            
            #Guardamos el modelo
            self.guardarModelo(clf)


    def getPalabrasNoticia(self, noticia):

        #Abrimos la noticia y guardamos en raw el texto plano
        f = open(noticia)
        raw = f.read()

        #Tokenizamos el texto guardandolo en una lista
        tokens = word_tokenize(raw)

        #Quitamos los elementos que no sean palabras
        filteredAlNum = [w.lower() for w in tokens if w.isalnum()]

        #Quitamos los elementos que sean preposiciones, determinantes, etc (palabras que no aportan la informacion que necesitamos)
        filteredStopwords = [w for w in filteredAlNum if not w in self.stop_words]

        #Eliminamos los sufijos y prefijos de las palabras de la ultima lista
        filteredStem = [self.stemmer.stem(w) for w in filteredStopwords]

        return filteredStem
    
    def getPalabrasContadas(self, listaNoticias, esDespoblacion):

        listaPalabrasContadas = []
        #Abrimos cada noticia y guardamos en un array las palabras de cada una
        for noticia in listaNoticias:
            listaPalabrasNoticia = self.getPalabrasNoticia(noticia)
            #Contamos todas las palabras que se repiten guardando las palabras y las veces que aparecen
            listaPalabrasContadasNoticia = Counter(listaPalabrasNoticia)
            #Añadimos una nueva pareja, que nos servira para tener este dato en el dataframe
            listaPalabrasContadasNoticia["esDespoblacion"] = esDespoblacion
            #print("lista palabras contadas")
            listaPalabrasContadas.append(listaPalabrasContadasNoticia)

        #print(listaPalabrasContadas)
        return listaPalabrasContadas


    
    def guardarModelo(self, model):
        filename = QFileDialog.getSaveFileName(self, caption="Guardar modelo", filter="Modelo (*.model)")
        with open(filename[0], 'wb') as file:
            pickle.dump(model, file)

    
    def cargarModelo(self, filename):
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)

        return pickle_model

    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()