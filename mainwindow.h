#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QPixmap>
#include <QTimer>
#include <QDebug>
#include <QGraphicsScene>

#include "neuralcluster.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_pushButton_clicked();
    void processNet();

    void on_pushButton_2_clicked();

private:
    vector<float> inputFunction(int type, int length, int periode, int phase);

private:
    Ui::MainWindow *ui;

    QImage *image;
    QImage *imageResp;

    bool running = false;

    int numLessons = 3;//sizeof (input)/sizeof (input[0]);
    int iteration = 0;
    float currentErrorBP,CurrentErrorMine;
    float lastErrorBP,lastErrorMine;

    int currentFrequency = 0;
    int phase = 0;
    int numInputs = 16;//sizeof (input[0])/sizeof (input[0][0]);
    int numOutputs = 3+8;//sizeof (output[0])/sizeof (output[0][0]);
    int numHiddens = 23;
    NeuralCluster* Cluster0;
    NeuralCluster* ClusterBP;
    QTimer *timer;
    QGraphicsScene* ErrorView;

};

#endif // MAINWINDOW_H