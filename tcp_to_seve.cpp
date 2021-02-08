#include "tcp_to_seve.h"

Tcp_to_seve::Tcp_to_seve(QObject *parent) : QThread(parent)
{
    sock = new QTcpSocket();
    sock->connectToHost(tcp_ip, tcp_port);
}


void Tcp_to_seve::run()
{
    //send datas file to severe teminal when received singal from severe teminal
    connect(sock, &QTcpSocket::readyRead, this, &Tcp_to_seve::sendDatas);
}


void Tcp_to_seve::sendDatas()
{
    QString filePath = "./related_file/personDatas.txt";

    quint64 len = 0;
    QFile file(filePath);
    bool isOK = file.open(QIODevice::ReadOnly);

    if(!isOK)
    {
       emit show_file_open_error();       //show error message after failed to open file
       return;
    }

    do{
       //每次发送数据的大小
       char buf[1024];
       //len = 0;
       //往文件中读数据
       len = file.read(buf,sizeof(buf));
       //发送数据,读多少，发多少
       len = sock->write(buf,len);

    }while(len > 0);

    file.close();
}




