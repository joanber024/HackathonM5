#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include <limits.h>
#include <omp.h>

#include "backprop.h"
#include "layer.h"
#include "common.h"


layer *lay = NULL;
int num_layers;
int *num_neurons;
float alpha;
float *cost;
float full_cost;
char **input;
int num_training_ex;
int n=1;
int total = 0;
int seed=50;
float tcost = 0;

//-----------RANDOM FUNCTIONS------------
int rando()
{
	seed = (214013*seed+2531011);
	return seed >>16;
}

float  random_between_two(float min, float max)    
{   
  return ((max - min) * ((float)rand() / RAND_MAX)) + min;
}

//-----------FREE INPUT------------
void freeInput( int np, char **input )
{
	for( int i = 0; i < np; i++ ) free( input[i] );
	free(input);
}

//-----------PRINTRECOGNIZED------------
void printRecognized( int p, layer Output ){
	int imax = 0;

	for( int i = 1; i < NUMOUT; i++)
		if ( Output.actv[i] > Output.actv[imax] ) imax = i;
	//printf( "El patró %d sembla un %c\t i és un %d", p, '0' + imax, Validation[p] );
	if( imax == Validation[p] ) total++;
  /*for( int k = 0 ; k < NUMOUT ; k++ )
 	  printf( "\t%f\t", Output[k].actv ) ;
  printf( "\n" );*/
}

//-----------INIT------------
int init()
{
  if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
  {
      printf("Error in creating architecture...\n");
      return ERR_INIT;
  }

  printf("Neural Network Created Successfully...\n\n");
  return SUCCESS_INIT;
}

//-----------DINIT------------
int dinit(void)
{
    // TODO:
    // Free up all the structures
    return SUCCESS_DINIT;
}

//-----------CREATE NEURAL NETWORK ARCHITECTURE------------
int create_architecture()
{
    int i=0,j=0;
    lay = (layer*) malloc(num_layers * sizeof(layer));

	 for(i=0;i<num_layers;i++)
    {   
      if (i < (num_layers - 1))
      {
        lay[i] = create_layer(num_neurons[i], num_neurons[i+1]);      
      }
      else
      {
        lay[i] = create_layer(num_neurons[i],0);  
      }       
    }

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        printf("Error Initilizing weights...\n");
        return ERR_CREATE_ARCHITECTURE;
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}


//-----------INITIALIZE WEIGHTS------------
int initialize_weights(void)
{
    if(lay == NULL)
    {
        printf("No layers in Neural Network...\n");
        return ERR_INIT_WEIGHTS;
    }

    printf("Initializing weights...\n");
    for(int i=0;i<num_layers-1;i++)
    {      
      for(int j=0;j<num_neurons[i];j++) 
      {
        for(int k=0;k<num_neurons[i+1];k++)
        {
            // Initialize Output Weights for each neuron                                
            lay[i].out_weights[k*num_neurons[i]+j] = random_between_two(-sqrt((float)2/(float)num_neurons[i]), sqrt((float)2/(float)num_neurons[i]));
    
            lay[i].dw[k*num_neurons[i]+j] = 0.0;
        }


        if(i>0)
        {				
          lay[i].bias[j] = 0;// random_between_two(-sqrt((float)2/(float)num_neurons[i-1]), sqrt((float)2/(float)num_neurons[i-1])); bias = 0 provides better solution				
        }
      }
    }
    	
    for (int j=0; j<num_neurons[num_layers-1]; j++)
    {
        lay[num_layers-1].bias[j] = 0;//random_between_two(-sqrt((float)2/(float)num_neurons[num_layers-2]), sqrt((float)2/(float)num_neurons[num_layers-2]));
        //((double)rand())/((double)RAND_MAX);        
    }

    return SUCCESS_INIT_WEIGHTS;
}

//-----------FEED INPUTS TO INPUT LAYER------------
void feed_input(int i)
{
	for(int j=0;j<num_neurons[0];j++)
	{
		lay[0].actv[j] = input[i][j];
	}	
}

//-----------FORWARD PROP------------
void forward_prop()
{
	for(int i=1;i<num_layers;i++)
	{
		for(int j=0;j<num_neurons[i];j++)
		{	
			lay[i].z[j] = lay[i].bias[j];      
			for(int k=0;k<num_neurons[i-1];k++) 
			{               
				lay[i].z[j] += ((lay[i-1].out_weights[j*num_neurons[i-1] + k])* (lay[i-1].actv[k]));
			}
      			
			// Relu Activation Function for Hidden Layers
			if(i< num_layers-1)
			{				
				if((lay[i].z[j]) < 0)
				{
					lay[i].actv[j] = 0;
				}

				else
				{
					lay[i].actv[j] = lay[i].z[j];
				}
			}
			else
			{
				lay[i].actv[j] = 1/(1+exp(-lay[i].z[j]));
				
			}				
		}
	}
}

//-----------COMPUTE TOTAL COST------------
void compute_cost(int i)
{
  float tmpcost=0;

  for(int j=0;j<num_neurons[num_layers-1];j++)
  {
      tmpcost = desired_outputs[i][j] - lay[num_layers-1].actv[j];
      cost[j] = (tmpcost * tmpcost)/2;
      tcost+= cost[j];
  }
  full_cost = (full_cost + tcost)/n;
  n++;
  printf("Full Cost: %f\n",full_cost);
}

//-----------BACK PROPAGATE ERROR------------
void back_prop(int p)
{
    
  // Output Layer
	for(int j=0;j<num_neurons[num_layers-1];j++)
	{           
		lay[num_layers-1].dz[j] = (lay[num_layers-1].actv[j] - desired_outputs[p][j]) * (lay[num_layers-1].actv[j]) * (1- lay[num_layers-1].actv[j]);
		lay[num_layers-1].dbias[j] = lay[num_layers-1].dz[j];           
	}

  for(int j=0;j<num_neurons[num_layers-1];j++)
	{      
		for(int k=0;k<num_neurons[num_layers-2];k++)
		{   
			lay[num_layers-2].dw[j*num_neurons[num_layers-2] + k] = (lay[num_layers-1].dz[j] * lay[num_layers-2].actv[k]);
			lay[num_layers-2].dactv[k] = lay[num_layers-2].out_weights[j*num_neurons[num_layers-2] + k] * lay[num_layers-1].dz[j];
		}
	}
  	
  // Hidden Layers
	for(int i=num_layers-2;i>0;i--)
	{
		for(int j=0;j<num_neurons[i];j++)
		{
			if(lay[i].z[j] >= 0)
			{
				lay[i].dz[j] = lay[i].dactv[j];
			}
			else
			{
				lay[i].dz[j] = 0;
			}

			for(int k=0;k<num_neurons[i-1];k++)
			{
				lay[i-1].dw[j*num_neurons[i-1] + k] = lay[i].dz[j] * lay[i-1].actv[k];
				
				if(i>1)
				{
					lay[i-1].dactv[k] = lay[i-1].out_weights[j* num_neurons[i-1] + k] * lay[i].dz[j];
				}
			}

			lay[i].dbias[j] = lay[i].dz[j];
		}
	}
}

//-----------UPDATE WEIGHTS------------
void update_weights(void)
{
	for(int i=0;i<num_layers-1;i++)
  {	
    for(int j=0;j<num_neurons[i+1];j++)
    {			
        for(int k=0;k<num_neurons[i];k++)
        {
          // Update Weights
          lay[i].out_weights[j*num_neurons[i]+ k] = (lay[i].out_weights[j*num_neurons[i]+ k]) - (alpha * lay[i].dw[j*num_neurons[i]+ k]);                
        }
    }
    
    for(int j=0;j<num_neurons[i];j++)
    {			
      // Update Bias
      lay[i].bias[j] = lay[i].bias[j] - (alpha * lay[i].dbias[j]);

    }
   }   
}

//-----------GET TRAINING EXAMPLES------------
void  get_data()
{    
    if( (input = loadPatternSet( 1934, "optdigits.tra", 1 ) ) == NULL){
        printf( "Loading Patterns: Error!!\n" );
		exit( -1 );
	  }
}

//-----------TRAIN NEURAL NETWORK------------
void train_neural_net(void)
{
  printf("Training Neural Network...\n");
	int ranpat[NUMPAT];
 
  // Gradient Descent
	for(int it=0;it<100;it++)
	{	
    //Train patterns randomly
		for(int p = 0; p < NUMPAT; p++)
		{
			ranpat[p] = p;
		}
    for(int p = 0; p < NUMPAT; p++)
    {	
      int x = rando();
      int np = (x * x) % NUMPAT;
      int op = ranpat[p]; ranpat[p] = ranpat[np]; ranpat[np] = op;
    }
	}
	    
	for(int i=0;i<num_training_ex;i++)
	{			
		int p = ranpat[i];
	
		feed_input(p);
		forward_prop();				
		compute_cost(p);				
		back_prop(p);
		update_weights();		
	}
  freeInput( NUMPAT, input );
  printf( "END TRAINING\n" );
}


//-----------TEST THE TRAINED NETWORK------------
void test_nn(void) 
{
    
  char **rSet;
  char *fname[NUMRPAT];
    
  if( (rSet = loadPatternSet( NUMRPAT, "optdigits.cv", 0 )) == NULL){
  printf( "Error!!\n" );
  exit( -1 );
  }
    
  for(int i=0; i<NUMRPAT; i++)
  {
    for(int j=0;j<num_neurons[0];j++)
    {
        lay[0].actv[j] = rSet[i][j];
    }

    forward_prop();   
    printRecognized( i, lay[num_layers-1]);
  }

  printf( "\nTotal encerts = %d\n", total );
  freeInput( NUMRPAT, rSet );
  printf("END TEST\n");
}

//-----------MAIN------------

int main(void)
{
    int i;
    clock_t start = clock();
    printf("Creating layers in Neural Network:\n");
    fflush(stdout);
    
    //Number of layers and memory allocation
    num_layers = 3;
    num_neurons = (int*) malloc(num_layers * sizeof(int));
    
    //Number of neurons per layer
    num_neurons[0] = 1024;
    num_neurons[1] = 117;
    num_neurons[2] = 10;

    // Initialize the neural network module
    if(init()!= SUCCESS_INIT)
    {
      printf("Error in Initialization...\n");
      exit(0);
    }

    //Learning rate (indicates what portion of gradient should be used)
    alpha = 0.15;

    //Number of training examples
    num_training_ex = 1934;    
    
    cost = (float *) malloc(num_neurons[num_layers-1] * sizeof(float));

    // Get Training Examples  
    get_data();
    
    //Train
    train_neural_net();
    
    //Test
    test_nn();

    if(dinit()!= SUCCESS_DINIT)
    {
        printf("Error in Dinitialization...\n");
    }
  
    clock_t end = clock();
  	printf( "\n\nGoodbye! (%f sec)\n\n", (end-start)/(1.0*CLOCKS_PER_SEC) ) ;
 
    return 0;
}