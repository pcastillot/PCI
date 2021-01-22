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
    listaPalabrasDespoblacion = []
    listaPalabrasNoDespoblacion = []
    listaPalabrasSinMarcar = []
    listaPalabrasContadasDespoblacion = []
    listaPalabrasContadasNoDespoblacion = []
    listaPalabrasContadasSinMarcar = []


    def __init__(self, *args, **kwargs):
        QtWidgets.QMainWindow.__init__(self, *args, **kwargs)
        self.setupUi(self)
        self.btnRutaDespoblacion.clicked.connect(lambda: self.abrirRuta(0))
        self.btnRutaNoDespoblacion.clicked.connect(lambda: self.abrirRuta(1))
        self.btnRutaSinMarcar.clicked.connect(lambda: self.abrirRuta(2))
        self.btnRutaNoticias.clicked.connect(lambda: self.abrirRuta(3))
        self.btnRutaModelo.clicked.connect(self.abrirModelo)
        self.btnGenerarModelo.clicked.connect(self.generarModelo)
        
        
    def abrirRuta(self, i):
        my_dir = str(QFileDialog.getExistingDirectory(self, "Abre una carpeta", expanduser("~"), QFileDialog.ShowDirsOnly))
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

            for noticia in self.noticiasNoDespoblacion:
                f = open(noticia)
                raw = f.read()
                tokens = word_tokenize(raw)
                filtered = [w for w in tokens if w.isalnum()]
                self.listaPalabrasNoDespoblacion += filtered
            self.listaPalabrasContadasNoDespoblacion = Counter(self.listaPalabrasNoDespoblacion)
            print(self.listaPalabrasContadasNoDespoblacion)


    
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()