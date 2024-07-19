#include "phi_layers.h"

#ifndef _PHI_UTILS
#define _PHI_UTILS


#define FLOAT_CONFIG_PARAMS_NUM 2
#define INT_CONFIG_PARAMS_NUM 6

PhiModel* read_model(char *filename);
PhiModelRunState* create_run_state(PhiConfig* config, unsigned int total_seq_len);
PhiModelInput* create_input();
float* read_vector(char* filename, unsigned int size);




#endif