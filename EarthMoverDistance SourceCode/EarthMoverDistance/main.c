/* example2.c */

#include <stdio.h>
#include<string.h>
#include <stdlib.h>

#include "emd.h"

float _COST[5000][5000];

float dist(feature_t *F1, feature_t *F2) { return _COST[*F1][*F2]; }

int main()
{
    char filename[] = "/Users/GongLi/PycharmProjects/ImageRecognition/EMDone/groundDistance";
    FILE *file = fopen ( filename, "r" );

    int X;
    int Y;

    if (file != NULL) {
        char line [100];

        // Read the first line
        fgets(line,sizeof line,file);

        char *p;
        p = strtok(line, " ");

        if(p)
            X = atoi(p);

        p = strtok(NULL, " ");

        if(p)
            Y = atoi(p);

        int row = 0;
        int column = 0;
        float value;

        while(fgets(line,sizeof line,file)!= NULL) /* read a line from a file */ {
            value = atof(line);

            _COST[row][column] = value;

            column ++;
            if(column == Y){
                row ++;
                column = 0;
            }

        }

        fclose(file);
    }

    feature_t f1[X];
    feature_t  f2[Y];
    int index = 0;

    for (; index < X; index ++)
        f1[index] = index;

    for(index = 0; index < Y; index ++)
        f2[index] = index;

    float weight = 1.0 / X;
    float w1[X];
    for(index = 0; index < X; index ++)
        w1[index] = weight;

    weight = 1.0 / Y;
    float  w2[Y];
    for(index = 0; index < Y; index ++)
        w2[index] = weight;

    signature_t s1 = {X, f1, w1}, s2 = {Y, f2, w2};

    float e;
    flow_t flow[X + Y - 1];
    int i , flowSize;

    e = emd(&s1, &s2, dist, flow, &flowSize);
//    printf("emd=%f\n", e);

    // Write distance into a file
    char buf[48];
    snprintf (buf, sizeof(buf), "%f", e);

    file = fopen ("/Users/GongLi/PycharmProjects/ImageRecognition/EMDone/result", "w" );
    fputs(buf, file);
    fputs("\n", file);


//    // write flow into result file
//    char str[500];
//    char temp[100];
//    for(int j = 0; j < X + Y - 1; j ++){
//        if (flow[j].amount > 0) {
//            sprintf(temp, "%d\t", flow[j].from);
//            strcpy(str, temp);
//
//            sprintf(temp, "%d\t", flow[j].to);
//            strcat(str, temp);
//
//            sprintf(temp, "%f\n", flow[j].amount);
//            strcat(str, temp);
//
//            fputs(str, file);
//        }
//    }
//    file->_close;

    return 0;
}