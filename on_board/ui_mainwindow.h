/********************************************************************************
** Form generated from reading UI file 'mainwindow.ui'
**
** Created by: Qt User Interface Compiler version 5.11.3
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QLabel>
#include <QtWidgets/QMainWindow>
#include <QtWidgets/QTextBrowser>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_MainWindow
{
public:
    QWidget *centralWidget;
    QLabel *dateLabel;
    QLabel *camera;
    QLabel *timeLabel;
    QLabel *iconLabel;
    QLabel *temLabel;
    QLabel *backLabel1;
    QLabel *titleLabel;
    QTextBrowser *describeText;
    QLabel *backgroundLabel;

    void setupUi(QMainWindow *MainWindow)
    {
        if (MainWindow->objectName().isEmpty())
            MainWindow->setObjectName(QStringLiteral("MainWindow"));
        MainWindow->resize(640, 480);
        centralWidget = new QWidget(MainWindow);
        centralWidget->setObjectName(QStringLiteral("centralWidget"));
        dateLabel = new QLabel(centralWidget);
        dateLabel->setObjectName(QStringLiteral("dateLabel"));
        dateLabel->setGeometry(QRect(540, 0, 101, 30));
        QFont font;
        font.setPointSize(12);
        dateLabel->setFont(font);
        camera = new QLabel(centralWidget);
        camera->setObjectName(QStringLiteral("camera"));
        camera->setGeometry(QRect(50, 100, 241, 291));
        timeLabel = new QLabel(centralWidget);
        timeLabel->setObjectName(QStringLiteral("timeLabel"));
        timeLabel->setGeometry(QRect(330, 100, 221, 61));
        QFont font1;
        font1.setPointSize(40);
        font1.setBold(true);
        font1.setWeight(75);
        timeLabel->setFont(font1);
        iconLabel = new QLabel(centralWidget);
        iconLabel->setObjectName(QStringLiteral("iconLabel"));
        iconLabel->setGeometry(QRect(0, 0, 110, 30));
        temLabel = new QLabel(centralWidget);
        temLabel->setObjectName(QStringLiteral("temLabel"));
        temLabel->setGeometry(QRect(340, 160, 151, 31));
        QFont font2;
        font2.setPointSize(21);
        font2.setBold(true);
        font2.setWeight(75);
        temLabel->setFont(font2);
        backLabel1 = new QLabel(centralWidget);
        backLabel1->setObjectName(QStringLiteral("backLabel1"));
        backLabel1->setGeometry(QRect(0, 0, 640, 30));
        titleLabel = new QLabel(centralWidget);
        titleLabel->setObjectName(QStringLiteral("titleLabel"));
        titleLabel->setGeometry(QRect(240, 0, 191, 30));
        titleLabel->setFont(font);
        describeText = new QTextBrowser(centralWidget);
        describeText->setObjectName(QStringLiteral("describeText"));
        describeText->setGeometry(QRect(340, 210, 241, 171));
        QFont font3;
        font3.setPointSize(13);
        describeText->setFont(font3);
        describeText->setStyleSheet(QLatin1String("background-color: rgb(38, 69, 84)\n"
""));
        describeText->setFrameShape(QFrame::NoFrame);
        backgroundLabel = new QLabel(centralWidget);
        backgroundLabel->setObjectName(QStringLiteral("backgroundLabel"));
        backgroundLabel->setGeometry(QRect(0, 0, 640, 480));
        MainWindow->setCentralWidget(centralWidget);
        backLabel1->raise();
        backgroundLabel->raise();
        dateLabel->raise();
        camera->raise();
        timeLabel->raise();
        iconLabel->raise();
        temLabel->raise();
        titleLabel->raise();
        describeText->raise();

        retranslateUi(MainWindow);

        QMetaObject::connectSlotsByName(MainWindow);
    } // setupUi

    void retranslateUi(QMainWindow *MainWindow)
    {
        MainWindow->setWindowTitle(QApplication::translate("MainWindow", "MainWindow", nullptr));
        dateLabel->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        camera->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        timeLabel->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        iconLabel->setText(QString());
        temLabel->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        backLabel1->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
        titleLabel->setText(QApplication::translate("MainWindow", "\346\231\272\346\205\247\344\272\221\351\227\250\347\246\201\350\200\203\345\213\244\347\273\210\347\253\257", nullptr));
        describeText->setHtml(QApplication::translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:'Ubuntu'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">\344\275\277\347\224\250\350\257\264\346\230\216\357\274\232</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">1.\350\257\267\345\260\206\351\235\242\351\203\250\351\235\240\350\277\221\345\267\246\346\226\271\346\243\200\346\265\213\345\214\272\345\237\237</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bott"
                        "om:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">2.\347\255\211\345\276\205\346\243\200\346\265\213\344\270\216\350\257\206\345\210\253</p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">3.\350\257\267\345\260\206\346\211\213\346\224\276\345\205\245\346\270\251\345\272\246\347\233\221\346\265\213\345\214\272\345\237\237</p></body></html>", nullptr));
        backgroundLabel->setText(QApplication::translate("MainWindow", "TextLabel", nullptr));
    } // retranslateUi

};

namespace Ui {
    class MainWindow: public Ui_MainWindow {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_MAINWINDOW_H
