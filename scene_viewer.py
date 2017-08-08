import os
import sys
import pickle
import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt5 import QtGui, QtWidgets, QtCore


class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.initUI()

    def initUI(self):
        self.scene = None
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.pointCloud = self.init_cloud()
        
        self.openFile = QtWidgets.QPushButton('Load scene', self)
        self.openFile.clicked.connect(self.open_file)
        self.removeOutliers = QtWidgets.QPushButton('Remove outliers', self)
        self.removeOutliers.clicked.connect(self.remove_outliers)
        self.removeOutliers.setEnabled(False)
        self.saveFile = QtWidgets.QPushButton('Save scene', self)
        self.saveFile.clicked.connect(self.save_scene)

        self.matches = self.init_matches()

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(['Frames'])
        self.tree.setFixedWidth(150)
        self.tree.setColumnWidth(0, 145)
        self.tree.itemClicked.connect(self.draw)

        self.grid = QtGui.QGridLayout()
        self.grid.addWidget(self.openFile, 0, 0, 1, 1)
        self.grid.addWidget(self.removeOutliers, 0, 1, 1, 1)
        self.grid.addWidget(self.saveFile, 0, 2, 1, 1)
        self.grid.addWidget(self.tree, 1, 0, 4, 1)
        self.grid.addWidget(self.matches, 5, 0, 1, 1)
        self.grid.addWidget(self.pointCloud, 1, 1, 5, 5)

        self.setLayout(self.grid)

        self.setGeometry(100, 100, 800, 600)
        self.setWindowTitle('Point Cloud View')
        self.show()


    def open_file(self):
        file = QtGui.QFileDialog.getOpenFileName(self, 'Select the point cloud .pkl file.', self.dir_path, filter=("pkl (*.pkl)"))[0]
        self.scene = pickle.load(open(file, 'rb'))
        self.load_frames()
        self.removeOutliers.setEnabled(True)


    def load_frames(self):
        
        self.tree.clear()
        self.toplevel = QtWidgets.QTreeWidgetItem(self.tree)
        self.toplevel.setText(0, 'All')
        self.toplevel.setFlags(self.toplevel.flags() | QtCore.Qt.ItemIsTristate | QtCore.Qt.ItemIsUserCheckable)

        self.frames = self.scene.build_frames()

        for frame in self.frames.keys():
            item = QtWidgets.QTreeWidgetItem(self.toplevel)
            item.setText(0, 'frame {:d}'.format(frame))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(0, QtCore.Qt.Checked)
        
        self.draw()

    def draw(self):
        self.tree.expandAll()
        frame_count = self.toplevel.childCount()
        
        frames_to_draw = 0
        X = []
        x = []
        r = []
        c = []
        for i in range(frame_count):
            frame = self.toplevel.child(i)
            if frame.checkState(0) == 2:
                frames_to_draw += 1
                X.append(self.frames[i]['3D'])
                x.append(self.frames[i]['2D'])
                r.append(self.frames[i]['re'])
                c.append(self.frames[i]['ca'])
        
        if frames_to_draw:
            X = np.vstack(X)
            x = np.vstack(x)
            r = np.vstack(r)
            c = np.vstack(c)

            # camera colors, size
            c_size = 20
            N_cams = c.shape[0]
            R = np.linspace(0, 1., N_cams)
            G = np.linspace(0.5, 1., N_cams)
            B = R[::-1]
            c_colors = np.column_stack((R, G, B))
            if N_cams == 1:
                    c_colors = np.array([0, 0, 1])

        else:
            x = np.zeros((1, 2))
            r = np.zeros((1, 2))
            X = np.zeros((1, 3))
            c = np.zeros((1, 3))
            
            c_size = 1
            c_colors = np.zeros((1, 3))

        self.featurePlot.setData(x=x[:, 0], y=x[:, 1])
        self.rePlot.setData(x=r[:, 0], y=r[:, 1])
        self.cloudPlot.setData(pos=X, color=np.ones((X.shape[0], 3)), size=2, pxMode=True)
        self.camPlot.setData(pos=c, color=c_colors, size=c_size, pxMode=True)
        

    def init_cloud(self):
        w = gl.GLViewWidget(self)
        w.opts['distance'] = 1
        gx = gl.GLGridItem()
        w.addItem(gx)
        self.cloudPlot = gl.GLScatterPlotItem(pos=np.zeros(3), color=np.zeros((1, 3)), size=1, pxMode=True)
        self.camPlot = gl.GLScatterPlotItem(pos=np.zeros(3), color=np.array([0, 0, 0]), size=1, pxMode=True)
        w.addItem(self.cloudPlot)
        w.addItem(self.camPlot)
        return w

    def init_matches(self):
        w = pg.GraphicsLayoutWidget()
        w.setFixedSize(150, 150)
        w.setFixedWidth(150)
        plot = w.addPlot()
        plot.hideAxis('left')
        plot.hideAxis('bottom')
        self.featurePlot = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(0, 127, 255, 255))
        self.rePlot = pg.ScatterPlotItem(size=2, pen=pg.mkPen(None), brush=pg.mkBrush(255, 255, 0, 200))
        plot.addItem(self.featurePlot)
        plot.addItem(self.rePlot)
        return w

    def remove_outliers(self):
        x, visibility, points2D, N_frames = self.scene.pack()
        self.scene.unpack(x, remove_outliers = True, max_sd_dist=1)
        self.load_frames()

    def save_scene(self):
        file = QtGui.QFileDialog.getSaveFileName(self, 'Select the save .pkl file.', self.dir_path, filter=("pkl (*.pkl)"))[0]
        root, ext = os.path.splitext(file)
        path, name = os.path.split(root)
        self.scene.pickle_save(name=name, path=path, mode='scene')


def main():
    app = QtWidgets.QApplication([])
    ex = MainWindow()
    sys.exit(app.exec_())
    

if __name__ == '__main__':
    main()