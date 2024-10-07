#ifndef LAYER_H
#define LAYER_H

typedef struct layer_t
{
	int num_neu;
	float *actv;
	float *bias;
	float *z;
	float *dactv;
	float *dbias;
	float *dz;
	float *out_weights;
	float *dw;
} layer;

layer create_layer(int num_neurons, int number_of_neurons_next_layer);

#endif


