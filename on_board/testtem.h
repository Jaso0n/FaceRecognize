#ifndef TESTTEM_H
#define TESTTEM_H

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>


class TestTem
{
public:
    TestTem();

    double culTem();
};

#endif // TESTTEM_H
