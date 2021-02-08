#include "testtemper.h"

double testTem()
{
    int fd;
    fd = open("/dev/i2c-2", O_RDWR);
    if (fd < 0)
    {
        perror("open i2c controller");
        return 1;
    }

    ioctl(fd,I2C_TIMEOUT,5);
    ioctl(fd,I2C_RETRIES,10);
    unsigned char reg =0x07;
    unsigned char datas[2];

    int i = 0,te;
    int buf[6];
    while(i != 7)
    {
        struct i2c_msg msgs[2] = {
            {DHT12_ADDR,0,1,&reg},
            {DHT12_ADDR, I2C_M_RD, sizeof(datas), datas},
        };
        struct i2c_rdwr_ioctl_data rdat = {
            .msgs = msgs,
            .nmsgs = 2,
        };
        if (ioctl(fd, I2C_RDWR, &rdat) < 0)
        {
            perror("i2c rdwr failed\n");
            return 3;
        }

        /*
         * * 	int i;
         * * 	for (i = 0; i < sizeof(datas); i++)
         * * 		printf("%02x", datas[i]);
         * * 		printf("\n");
        */
        buf[i] = datas[1]<<8|datas[0];
        i = i+1;
        sleep(0.1);
        //printf("%d\n",temp0);
    }
    double a;
    te = (buf[0]+buf[1]+buf[2]+buf[3]+buf[4]+buf[5]+buf[6])/7;
    a=te*0.02-273.15+6.2+3.7-2.5;
    //printf("%3.2f\n",a);
    sleep(1);
    close(fd);
    return a;
}
