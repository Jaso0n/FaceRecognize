#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "workthread.h"
#include "tcp_to_seve.h"
#include "testtem.h"

#include <QMainWindow>
#include <QApplication>
#include <iostream>
#include <fstream>
#include <string>
#include <QPushButton>
#include <QLabel>
#include <QTimer>
#include <QImage>
#include <QDateTime>
#include <QDebug>
#include <QMessageBox>
#include <cv.h>
#include <highgui.h>



namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

    QImage cvMat2QImage(const Mat &src);

    void openCamara(); //打开时摄像头

    string ReadText(string filename, int line);

    void main_face_detection();
    
    WorkThread *t1;
    WorkThread *t2;


public slots:

    void stopThread();

    void inputImage();

    void showImage();

    void showCurrentTime();

    void coltrol_tcp_tem(int);    //识别出人脸后检测温度，Int为向量下标

    void show_file_error_message();



private:
    Ui::MainWindow *ui;

    Tcp_to_seve *clint;
    TestTem *tem;       //检测温度对象


    QImage imag;
    QImage *iconImage;
    QImage *backImage;

    int idxInputImage = 0;
    VideoCapture cap;

    QTimer *showTime;
    QTimer *getImageTimer;
    QTimer *showImageTimer;

};

#endif // MAINWINDOW_H
