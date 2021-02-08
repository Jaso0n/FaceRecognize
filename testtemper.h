#ifndef TESTTEMPER_H
#define TESTTEMPER_H

#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <linux/i2c.h>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>

#define DHT12_ADDR  0x5a

unsigned char data[2];

double testTem();

#endif // TESTTEMPER_H
