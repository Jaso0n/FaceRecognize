#ifndef TCP_TO_SEVE_H
#define TCP_TO_SEVE_H

#include <QObject>
#include <QThread>
#include <QTcpSocket>
#include <stdio.h>
#include <QString>
#include <QFile>

#define tcp_ip "192.168.124.6"
#define tcp_port 9090


class Tcp_to_seve : public QThread
{
    Q_OBJECT
public:
    explicit Tcp_to_seve(QObject *parent = nullptr);
    QTcpSocket *sock;

    void run();



signals:
    void show_file_open_error();


public slots:

    void sendDatas();
};

#endif // TCP_TO_SEVE_H
