#include "logistic_layer.h"
#include "blas.h"
#include "dark_cuda.h"
#include "utils.h"
#include "blas.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#define SECRET_NUM -1234


logistic_layer make_logistic_layer(int batch, int inputs, int groups, int stride, float *classes_multipliers, int classes)
{
    assert(inputs%groups == 0);
    fprintf(stderr, "logistic                                        %4d\n",  inputs);
    logistic_layer l = { (LAYER_TYPE)0 };
    l.type = LOGXENT;
    l.batch = batch;
    l.groups = groups;
    l.stride = stride;
    l.inputs = inputs;
    l.outputs = inputs;
    l.loss = (float*)xcalloc(inputs * batch, sizeof(float));
    l.output = (float*)xcalloc(inputs * batch, sizeof(float));
    l.delta = (float*)xcalloc(inputs * batch, sizeof(float));
    l.cost = (float*)xcalloc(1, sizeof(float));

    l.forward = forward_logistic_layer;
    l.backward = backward_logistic_layer;
    l.biases = NULL;
    if (classes_multipliers) {
      l.biases = classes_multipliers;
    }
#ifdef GPU
    l.forward_gpu = forward_logistic_layer_gpu;
    l.backward_gpu = backward_logistic_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch);
    l.loss_gpu = cuda_make_array(l.loss, inputs*batch);
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch);
    l.biases_gpu = NULL;
    if (l.biases) {
      l.biases_gpu = cuda_make_array(l.biases, classes);
    }
#endif
    return l;
}

void forward_logistic_layer(const logistic_layer l, network_state net)
{
  copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
  activate_array(l.output, l.outputs*l.batch, LOGISTIC);
  if(net.truth){
    logistic_x_ent_cpu(l.batch*l.inputs, l.output, net.truth, l.delta, l.loss);
    l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
  }  
}

void backward_logistic_layer(const logistic_layer l, network_state net)
{
    axpy_cpu(l.inputs*l.batch, 1, l.delta, 1, net.delta, 1);
}

#ifdef GPU

void pull_logistic_layer_output(const logistic_layer layer)
{
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs*layer.batch);
}

void forward_logistic_layer_gpu(const logistic_layer l, network_state net)
{
  int i, j, k;
  simple_copy_ongpu(l.batch*l.inputs, net.input, l.output_gpu);
  //  copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
  //  activate_array_gpu(l.output_gpu, l.outputs*l.batch, LOGISTIC);
  //  cuda_pull_array(l.output_gpu, net.input, l.batch*l.inputs);
  //cuda_pull_array(net.truth_gpu, net.truth, l.batch*l.inputs);
  //image im = make_image(1024, 512, 1);
  //image im_truth = make_image(1024, 512, 1);
  /*
  for(i=0; i<l.w*l.h; i++){
       l.delta[i] = 0 - net.input[i];
    //im.data[i] = (float)net.input[i];
    //im_truth.data[i] = (float)net.truth[i];
    }*/
  //cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
  //save_image(im, "feature_map");
  //save_image(im_truth, "truth");
  //free_image(im);
  //free_image(im_truth);
  if(net.truth){
    logistic_x_ent_gpu(l.batch*l.inputs, l.output_gpu, net.truth, l.delta_gpu, l.loss_gpu);
       cuda_pull_array(l.loss_gpu, l.loss, l.batch*l.inputs);
       


       /*
       cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
       image m = make_image(480, 480, 1);       
       m.data = l.output;
s
       save_image_png(m, "depth_esti");       
       cuda_pull_array(net.truth, l.output, l.batch*l.inputs);
       m.data = l.output;

       save_image_png(m, "depth_trainer");
       */
    //   cuda_pull_array(l.delta_gpu, l.loss, l.batch*l.inputs);    
    l.cost[0] = sum_array(l.loss, l.batch*l.inputs);
    l.cost[0] = mean_array(l.loss, l.batch*l.inputs);    
    printf("Logistic Regression Cost %lf\n", l.cost[0]);
    cuda_pull_array(l.output_gpu, l.output, l.batch*l.inputs);
    cuda_pull_array(net.truth, l.loss, l.batch*l.inputs);
    int test =1;
  }
}

void backward_logistic_layer_gpu(const logistic_layer layer, network_state state)
{
  axpy_ongpu(layer.batch*layer.inputs, state.net.loss_scale, layer.delta_gpu, 1, state.delta, 1);
}

#endif

