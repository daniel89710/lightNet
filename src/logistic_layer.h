#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H
#include "layer.h"
#include "network.h"

typedef layer logistic_layer;


#ifdef __cplusplus
extern "C" {
#endif
void logistic_array(float *input, int n, float temp, float *output);
logistic_layer make_logistic_layer(int batch, int inputs, int groups, int stride, float *class_multipliers, int classes);
void forward_logistic_layer(const logistic_layer l, network_state state);
void backward_logistic_layer(const logistic_layer l, network_state state);

#ifdef GPU
void pull_logistic_layer_output(const logistic_layer l);
void forward_logistic_layer_gpu(const logistic_layer l, network_state state);
void backward_logistic_layer_gpu(const logistic_layer l, network_state state);
#endif


#ifdef __cplusplus
}
#endif
#endif
