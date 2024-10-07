#include "layer.h"
#include <stdlib.h>
#include <stdio.h>

layer create_layer(int number_of_neurons, int number_of_neurons_next_layer)
{
	
	layer lay;
	lay.num_neu = number_of_neurons;
	
	lay.actv = malloc(number_of_neurons*sizeof(float));
	lay.bias = malloc(number_of_neurons*sizeof(float));
	lay.z = malloc(number_of_neurons*sizeof(float));
	lay.dactv = malloc(number_of_neurons*sizeof(float));
	lay.dbias = malloc(number_of_neurons*sizeof(float));
	lay.dz = malloc(number_of_neurons*sizeof(float));
	  

	lay.out_weights = malloc(number_of_neurons*number_of_neurons_next_layer * sizeof(float));
	lay.dw = malloc(number_of_neurons*number_of_neurons_next_layer * sizeof(float));
  
	return lay;
}
