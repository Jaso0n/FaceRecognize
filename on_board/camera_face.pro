#-------------------------------------------------
#
# Project created by QtCreator 2020-10-18T09:40:47
#
#-------------------------------------------------

QT       += core gui network


greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = camera_face
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0


SOURCES += \
        main.cpp \
        mainwindow.cpp \
    workthread.cpp \
    tcp_to_seve.cpp \
    testtem.cpp

HEADERS += \
        mainwindow.h \
    workthread.h \
    tcp_to_seve.h \
    testtem.h

FORMS += \
        mainwindow.ui

INCLUDEPATH+=/usr/include\
/usr/include/opencv\
/usr/include/opencv2\
/usr/local/include/dnndk
LIBS+=/usr/lib/libopencv_highgui.so\
/usr/lib/libopencv_core.so\
/usr/lib/libopencv_imgproc.so\
/usr/lib/libopencv_imgcodecs.so\
/usr/lib/libopencv_videoio.so\
/usr/local/lib/libdputils.so\
/usr/local/lib/libn2cube.so\
/usr/local/lib/libhineon.so\
./model/dpu_resnet50.elf\
./model/dpu_ultra96v2_DenseBox_360x640_2304.elf

RESOURCES += \
    persondatas.qrc \
    images.qrc
