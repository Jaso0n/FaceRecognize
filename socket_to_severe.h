#ifndef SOCKET_TO_SEVERE_H
#define SOCKET_TO_SEVERE_H

#include <QWidget>
#include <QTcpSocket>

class Socket_to_severe : public QWidget
{
    Q_OBJECT
public:
    explicit Socket_to_severe(QWidget *parent = nullptr);
    QTcpSocket *sock;

signals:

public slots:
};

#endif // SOCKET_TO_SEVERE_H
