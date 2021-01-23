from main_ui import *
from os import listdir
from os.path import expanduser, isfile, join
from PyQt5.QtWidgets import QFileDialog
import glob
import ctypes
from nltk import *
from collections import Counter




class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    dirModelo = None
    dirDespoblacion = "..\\Datos\\despoblacion"
    dirNoDespoblacion = "..\\Datos\\no_despoblacion"
    dirSinMarcar = "..\\Datos\\unlabeled"
    dirRutaNoticias = None
    noticiasDespoblacion = None
    noticiasNoDespoblacion = None
    noticiasSinMarcar = None
    noticiasTesting = None
    listaPalabrasContadasDespoblacion = []
    listaPalabrasContadasNoDespoblacion = []
    listaPalabrasContadasSinMarcar = []

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
    

    def getFicherosDirectorio(self, dir):
        ficheros = glob.glob(dir + "/*.txt")
        return ficheros


    def generarModelo(self):
        if self.dirDespoblacion == None or self.dirNoDespoblacion == None or self.dirSinMarcar == None:
            ctypes.windll.user32.MessageBoxW(0, "Debes elegir una ruta para todos las noticias", "Error al recuperar noticias", 0)
        else:
            self.noticiasNoDespoblacion = self.getFicherosDirectorio(self.dirNoDespoblacion)
            self.noticiasDespoblacion = self.getFicherosDirectorio(self.dirDespoblacion)
            self.noticiasSinMarcar = self.getFicherosDirectorio(self.dirSinMarcar)

            #Procesamiento de texto
            self.listaPalabrasContadasDespoblacion = self.getPalabrasContadas(self.noticiasDespoblacion)
            self.listaPalabrasContadasNoDespoblacion = self.getPalabrasContadas(self.noticiasNoDespoblacion)

            print("Palabras Despoblacion:")
            print(self.listaPalabrasContadasDespoblacion)
            print("---------------------------------")
            print("Palabras No Despoblacion:")
            print(self.listaPalabrasContadasNoDespoblacion)

    
    def getPalabrasContadas(self, listaNoticias):

        listaPalabras = []

        for noticia in listaNoticias:

            #Abrimos cada noticia de no despoblacion y guardamos en raw el texto plano
            print(noticia)
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

            #Añadimos a la lista de palabras que aparecen en las noticias de no despoblacion la ultima lista
            listaPalabras += filteredStem
        
        #Contamos todas las palabras que se repiten guardando en una tupla las palabras y las veces que aparecen
        palabrasContadas = Counter(listaPalabras)
        return palabrasContadas

    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()