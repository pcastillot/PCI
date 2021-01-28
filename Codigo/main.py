from main_ui import *
from os import listdir
from os.path import expanduser, isfile, join
from PyQt5.QtWidgets import QFileDialog, QTableWidgetItem
from nltk import *
from collections import Counter
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
from joblib import dump, load
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
import pickle
import pandas as pd
import glob
import ctypes
import numpy as np
import random

class Modelo():

    def __init__(self, clasificador, precision, acierto, listaPalabras, matrizResultados, algoritmo):
        self.clasificador = clasificador
        self.precision = precision
        self.acierto = acierto
        self.listaPalabras = listaPalabras
        self.matrizResultados = matrizResultados
        self.algoritmo = algoritmo


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    dirModelo = None
    dirDespoblacion = "..\\Datos\\despoblacion"
    dirNoDespoblacion = "..\\Datos\\no_despoblacion"
    dirRutaNoticias = "..\\Datos\\unlabeled"
    noticiasTesting = None
    listaPalabrasContadasDespoblacion = []
    listaPalabrasContadasNoDespoblacion = []
    listaPalabrasContadasSinMarcar = []
    listaPalabras = []
    modelo = None
    listaPalabrasCompleta = []
    listaNoticias = []
    modeloGenerado = None

    

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
        self.btnRutaNoticias.clicked.connect(lambda: self.abrirRuta(2))
        self.btnGuardarModelo.clicked.connect(lambda: self.guardarModelo(self.modeloGenerado))
        self.btnRutaModelo.clicked.connect(self.abrirModelo)
        self.btnGenerarModelo.clicked.connect(self.generarModelo)
        self.btnExportar.clicked.connect(lambda: self.exportarResultado(self.noticiasTesting))
        
        #Desactivado hasta solucionarlo
        #self.btnClasificar.clicked.connect(self.clasificarNoticias)

        listaTest = self.generarListaTest()
        listaPrediccion = self.generarPrediccionTest()
        self.btnClasificar.clicked.connect(lambda: self.mostrarResultadoNoticias(listaTest, listaPrediccion))
        
    
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
            self.lblRutaNoticias.setText(my_dir)
            self.dirRutaNoticias = my_dir

    def abrirModelo(self):
        #Obtenemos la ruta del modelo
        modelo = QFileDialog.getOpenFileName(self, 'Abrir modelo', expanduser("~"), "Modelo (*.model)")
        if modelo[0] != "":
            #Si hemos obtenido una ruta, cargamos el modelo en la variable y mostramos las estadisticas
            self.lblRutaModelo.setText(modelo[0])
            self.dirModelo = modelo[0]
            self.modelo = self.cargarModelo(self.dirModelo)
            self.lblAlgoritmoTesting.setText(self.modelo.algoritmo)
            self.lblAciertoTesting.setText(str(self.modelo.acierto) + "%")
            self.lblPrecisionTesting.setText(str(self.modelo.precision) + "%")
            self.mostrarResultadosTabla(self.modelo.matrizResultados, self.tableEstadisticasTesting)
    

    def getFicherosDirectorio(self, dir):
        #Crea una lista con la ruta de todos los ficheros
        ficheros = glob.glob(dir + "/*.txt")
        return ficheros


    def generarModelo(self):
        #Comprobacion de que todas las rutas se han seleccionado
        if self.dirDespoblacion == None or self.dirNoDespoblacion == None:
            #Mensaje de error
            ctypes.windll.user32.MessageBoxW(0, "Debes elegir una ruta para todos las noticias", "Error al recuperar noticias", 0)
        else:
            #Obtenemos las rutas de los archivos
            noticiasNoDespoblacion = self.getFicherosDirectorio(self.dirNoDespoblacion)
            noticiasDespoblacion = self.getFicherosDirectorio(self.dirDespoblacion)

            #Procesamiento de texto
            listaPalabrasContadasDespoblacion = self.getPalabrasContadas(noticiasDespoblacion, 1)
            listaPalabrasContadasNoDespoblacion = self.getPalabrasContadas(noticiasNoDespoblacion, 0)

            #print("Lista Palabras Contadas Despoblacion")
            #print(listaPalabrasContadasDespoblacion)
            #print("Lista Palabras Contadas No Despoblacion")
            #print(listaPalabrasContadasNoDespoblacion)

            listaPalabrasContadas = listaPalabrasContadasDespoblacion + listaPalabrasContadasNoDespoblacion

            #print("Lista Palabras Contadas")
            #print(listaPalabrasContadas)


            #Obtenemos el counter total para guardar todas las palabras en un array
            counterTotal = Counter()
            for counter in listaPalabrasContadas:
                counterTotal += counter
            
            #print("Counter Total:")
            #print(counterTotal)


            listaPalabras = []
            for palabra in counterTotal:
                listaPalabras.append(palabra)
            
            self.listaPalabrasCompleta = listaPalabras

            #print("listaPalabras")
            #print(listaPalabras)            

            #Creamos el dataframe que le pasaremos al modelo con todas las palabras en las columnas y una fila por noticia 
            df = pd.DataFrame(listaPalabrasContadas, columns=listaPalabras).fillna(0)
            print(df)

            #En X tenemos todas las palabras y su cuenta, es decir los datos que vamos a necesitar para predecir
            X = df.drop("esDespoblacion", axis=1)

            #En y tenemos la columna que queremos que prediga
            y = df["esDespoblacion"]


            #Divide los datos en training y testing
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) # 80% training y 20% test

            #Creamos el clasificador dependiendo del algoritmo elegido y lo entrenamos
            #Arbol de decision
            if self.cbAlgoritmo.currentIndex() == 0:
                clf = DecisionTreeClassifier()    
                clf = clf.fit(X_train,y_train)
            
            #K-nn Neighbours
            elif self.cbAlgoritmo.currentIndex() == 1:
                clf = KNeighborsClassifier(n_neighbors=5)   
                clf = clf.fit(X_train,y_train)

            #Naive Bayes
            elif self.cbAlgoritmo.currentIndex() == 2:
                clf = MultinomialNB()   
                clf = clf.fit(X_train,y_train)

            #Algoritmo 4
            elif self.cbAlgoritmo.currentIndex() == 3:
                clf = DecisionTreeClassifier()    
                clf = clf.fit(X_train,y_train)

            #Algoritmo 5
            elif self.cbAlgoritmo.currentIndex() == 4:
                clf = DecisionTreeClassifier()    
                clf = clf.fit(X_train,y_train)

            #Guardamos en variables los resultados de la prediccion
            y_pred = clf.predict(X_test)

            algoritmo = self.cbAlgoritmo.currentText()
            acierto = round(metrics.accuracy_score(y_test, y_pred)*100, 2)
            precision = round(metrics.precision_score(y_test, y_pred)*100, 2)
            

            #Imprimimos el algoritmo utilizado en la etiqueta
            self.lblAlgoritmo.setText(algoritmo)

            #Imprimimos el % de acierto en la etiqueta
            self.lblAcierto.setText(str(acierto) + "%")

            #Imprimimos el % de precision en la etiqueta
            self.lblPrecision.setText(str(precision) + "%")
            
            #Obtenemos la matriz de resultados e imprimimos resultados en la tabla
            matrizResultados = confusion_matrix(y_test, y_pred)

            self.mostrarResultadosTabla(matrizResultados, self.tableEstadisticas)
            
            #Guardamos el modelo generado
            self.modeloGenerado = Modelo(clf, precision, acierto, listaPalabras, matrizResultados, algoritmo)

    

    def mostrarResultadosTabla(self, matrizResultados, tableEstadisticas):
        tableEstadisticas.setItem(0, 0, QTableWidgetItem(str(matrizResultados[0][0])))
        tableEstadisticas.setItem(0, 1, QTableWidgetItem(str(matrizResultados[0][1])))
        tableEstadisticas.setItem(1, 0, QTableWidgetItem(str(matrizResultados[1][0])))
        tableEstadisticas.setItem(1, 1, QTableWidgetItem(str(matrizResultados[1][1])))
        tableEstadisticas.setItem(2, 0, QTableWidgetItem(str(round(matrizResultados[0][0]/(matrizResultados[0][0]+matrizResultados[1][0]), 2)*100) + "%"))
        tableEstadisticas.setItem(2, 1, QTableWidgetItem(str(round(matrizResultados[1][1]/(matrizResultados[0][1]+matrizResultados[1][1]), 2)*100) + "%"))
        tableEstadisticas.setItem(0, 2, QTableWidgetItem(str(round(matrizResultados[0][0]/(matrizResultados[0][0]+matrizResultados[0][1]), 2)*100) + "%"))
        tableEstadisticas.setItem(1, 2, QTableWidgetItem(str(round(matrizResultados[1][1]/(matrizResultados[1][0]+matrizResultados[1][1]), 2)*100) + "%"))

    def clasificarNoticias(self):

        #Si tenemos un modelo y ruta seleccionados comenzamos el procesamiento de las noticias
        if self.modelo and self.dirRutaNoticias:
            noticias = self.getFicherosDirectorio(self.dirRutaNoticias)
            listaPalabrasContadas = self.getPalabrasContadas(noticias)

            counterTotal = Counter()
            for counter in listaPalabrasContadas:
                counterTotal += counter

            listaPalabras = []
            for palabra in counterTotal:
                listaPalabras.append(palabra)
            
            listaPalabras += self.listaPalabrasCompleta

            listaPalabrasFinal = []

            for palabra in listaPalabras:
                if palabra not in listaPalabrasFinal:
                    listaPalabrasFinal.append(palabra)
            
            dfNoticias = pd.DataFrame(listaPalabrasContadas, columns=listaPalabrasFinal).fillna(0)

            X = dfNoticias

            y_pred = self.modelo.predict(X)
            print(y_pred)

            #Creamos un diccionario con una clave por noticia y su valor sera la prediccion
            for noticia in noticias:
                self.noticiasTesting[noticia] = y_pred[noticias.index(noticia)]
            
            self.mostrarResultadoNoticias(self.noticiasTesting)
        
        else:
            if self.modelo == None:
                ctypes.windll.user32.MessageBoxW(0, "Debes elegir un modelo", "Error al recuperar modelo", 0)
            
            if self.dirRutaNoticias == None:
                ctypes.windll.user32.MessageBoxW(0, "Debes elegir una ruta para las noticias", "Error al recuperar noticias", 0)
            




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
    
    def getPalabrasContadas(self, listaNoticias, esDespoblacion=None):

        listaPalabrasContadas = []
        #Abrimos cada noticia y guardamos en un array las palabras de cada una
        for noticia in listaNoticias:
            listaPalabrasNoticia = self.getPalabrasNoticia(noticia)
            #Contamos todas las palabras que se repiten guardando las palabras y las veces que aparecen
            listaPalabrasContadasNoticia = Counter(listaPalabrasNoticia)
            if esDespoblacion:
                #Añadimos una nueva pareja, que nos servira para tener este dato en el dataframe
                listaPalabrasContadasNoticia["esDespoblacion"] = esDespoblacion
            #print("lista palabras contadas")
            listaPalabrasContadas.append(listaPalabrasContadasNoticia)

        #print(listaPalabrasContadas)
        return listaPalabrasContadas

    
    def exportarResultado(self, dicNoticias):
        if dicNoticias:
            for noticia in dicNoticias:
                if dicNoticias[noticia]==1:
                    #Mover noticia a carpeta prediccion/despoblacion
                    print("Moviendo a despoblacion: " + noticia)
                else:
                    #Mover noticia a carpeta prediccion/no_despoblacion
                    print("Moviendo a no despoblacion: " + noticia)
        
        else:
            ctypes.windll.user32.MessageBoxW(0, "Debes clasificar las noticias con un modelo primero", "Error al exportar resultado", 0)


    
    def guardarModelo(self, modelo):
        if modelo:
            filename = QFileDialog.getSaveFileName(self, caption="Guardar modelo", filter="Modelo (*.model)")
            if filename[0] != "":
                with open(filename[0], 'wb') as file:
                    pickle.dump(modelo, file)
        else:
            ctypes.windll.user32.MessageBoxW(0, "Debes generar un modelo", "Error al guardar modelo", 0)

    
    def cargarModelo(self, filename):
        with open(filename, 'rb') as file:
            pickle_model = pickle.load(file)

        return pickle_model

    def mostrarResultadoNoticias(self, listaNoticias, listaPrediccion):
        #Inicializamos las variables y establecemos el numero de filas
        i = 0
        prediccion = "No despoblacion"
        self.tableResultados.setRowCount(len(listaNoticias))

        #Por cada noticia se muestra en la tabla la noticia y la prediccion
        for noticia in listaNoticias:
            self.tableResultados.setItem(i, 0, QTableWidgetItem(listaNoticias[i]))
            if listaPrediccion[i] == 1:
                prediccion = "Despoblacion"

            self.tableResultados.setItem(i, 1, QTableWidgetItem(prediccion))
            i += 1
    




    #Funciones para testear
    def generarListaTest(self):
        listaTest = []
        for i in range(50):
            listaTest.append("Noticia " + str(i))
        
        return listaTest

    def generarPrediccionTest(self):
        listaPrediccion = []
        for i in range(50):
            listaPrediccion.append(random.randrange(2))
        
        return listaPrediccion


    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()