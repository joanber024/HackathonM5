/*
 *  config.c
 *
 *  Created on: March 23rd 2022
 *  Author: ecesar asikora
 *
 *  Description:
 *  Functions for reading the program configuration file.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

//TOTES AQUESTES HAN D'ANAR A ALTRE ARXIUS
#define EXIT_FAILURE 1
int num_layers;
int *neurons_by_layer;
float alpha;
int batch_size;
int num_epochs;
int num_train_imgs;
int num_val_imgs;
int img_dim_x, img_dim_y;
char dataset_name[256];

// Igual pot anar a utils
void checkError(int ok, char *msg, char *file){
    if ( !ok ) {
        if( file == NULL) fprintf(stderr, "-- Error: %s\n", msg);
        else fprintf(stderr, "-- Error: %s File: %s\n", msg, file);
        exit( EXIT_FAILURE );
    }    
}

void printConfiguration(){
    printf("%d ", num_layers);
    for( int i=0; i < num_layers; i++ ) printf("%d ", neurons_by_layer[i]);
    printf("\n");

    printf("%f\n", alpha);
    printf("%d\n", batch_size);
    printf("%d\n", num_epochs);
    printf("%d\n", num_train_imgs);
    printf("%d\n", num_val_imgs);
    printf("%d %d\n", img_dim_x, img_dim_y);
    printf("%s\n", dataset_name);
}

void readConfiguration(char *configfile){
    int ok;

    FILE *fd = fopen(configfile, "r");
    checkError(fd!=NULL, "file not found", configfile);

    ok = fscanf(fd, "%d", &num_layers);
    checkError(ok!=EOF, "reading num_layers", configfile);
    
    neurons_by_layer = malloc(num_layers*sizeof(int));
    checkError(neurons_by_layer!=NULL, "allocating neurons_by_layer\n", NULL);
   
    for (int i = 0; i < num_layers; i++){
        ok = fscanf(fd, "%d", &neurons_by_layer[i]);
        checkError(ok!=EOF, "reading num_layers", configfile);
    }

    ok = fscanf(fd, "%f", &alpha);
    checkError(ok!=EOF, "reading alpha", configfile);
    
    ok = fscanf(fd, "%d", &batch_size);
    checkError(ok!=EOF, "reading batch_size", configfile);

    ok = fscanf(fd, "%d", &num_epochs);
    checkError(ok!=EOF, "reading num_epochs", configfile);
    
    ok = fscanf(fd, "%d", &num_train_imgs);
    checkError(ok!=EOF, "reading num_train_imgs", configfile);

    ok = fscanf(fd, "%d", &num_val_imgs);
    checkError(ok!=EOF, "reading num_val_imgs", configfile);

    ok = fscanf(fd, "%d %d", &img_dim_x, &img_dim_y);
    checkError(ok!=EOF, "reading image dimensions", configfile);

    ok = fscanf(fd, "%s", dataset_name);
    checkError(ok!=EOF, "reading dataset_name", configfile);
}