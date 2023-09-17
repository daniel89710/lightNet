/*
 * ----------------------------------------------------------------------------
 * "THE BEER-WARE LICENSE" (Revision 42):
 * Dan Umeda wrote this code. As long as you retain this notice you
 * can do whatever you want with this stuff. If we meet some day, and you think
 * this stuff is worth it, you can buy me a beer in return. - Dan Umeda
 *
 * This code is based on the work of Joseph Redmon and Alexey Bochkovskiy,
 * the original authors of YOLO. Their contributions to the field of computer
 * vision and deep learning are greatly appreciated, and this work would not
 * have been possible without them.
 * ----------------------------------------------------------------------------
 */

#include <stdlib.h>
#include "darknet.h"
#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"
#include "db.h"


#ifndef __COMPAR_FN_T
#define __COMPAR_FN_T
typedef int (*__compar_fn_t)(const void*, const void*);
#ifdef __USE_GNU
typedef __compar_fn_t comparison_fn_t;
#endif
#endif

#include "http_stream.h"

static int check_mistakes = 0;

static int coco_ids[] = { 1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90 };
extern float validate_segmenter_iou(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net, sqlite3 *db);

static void show_metrics(Eval_segmentation *eval, int classes, char *filename, char **names)
{
  int c;
  printf("##%s =>", filename);
  float iou, recall, precision;
  for (c = 0; c < classes; c++) {
    if ((eval[c].tp + eval[c].fn + eval[c].fp) == 0) {
      iou = 0.0;
    } else {
      iou = (eval[c].tp)/(float)(eval[c].tp + eval[c].fn + eval[c].fp + 0.00000001);
    }
    if ((eval[c].tp + eval[c].fn + eval[c].fp) == 0) {
      recall = 0.0;
    } else {
      recall = (eval[c].tp)/(float)(eval[c].tp + eval[c].fn + 0.00000001);
    }
    if ((eval[c].tp + eval[c].fn + eval[c].fp) == 0) {
      precision = 0.0;
    } else {
      precision = (eval[c].tp)/(float)(eval[c].tp + eval[c].fp + 0.00000001);
    }
    printf("(%s iou=%f,recall=%f,precision=%f) ", names[c], iou, recall, precision);
  }
  printf("\n");
}

static Eval_segmentation *evaluate_segmentation(unsigned char *pr, float *gt, int width, int height, int classes)
{  
  int x, y, c;
  Eval_segmentation *eval = (Eval_segmentation *)malloc(sizeof(Eval_segmentation) * classes);
  for (c = 0; c < classes; c++) {
    eval[c].tp = 0;
    eval[c].fn = 0;
    eval[c].fp = 0;          
  }  
  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      int index = y * width + x;
      int truth = (int)(gt[index] * 255.0);
      int pred = (int)pr[index];
      if (pred >= classes) {
	printf("Irregular pred value %d\n", pred);
      }
      if (truth >= classes) {
	printf("Irregular truth value %d\n", truth);
      }      
      if (truth == pred) {
	eval[truth].tp++;
      } else {
	eval[truth].fn++;
	eval[pred].fp++;
      }
    }
  }
  return eval;
}

static unsigned char* apply_argmax(float *output, int width, int height, int chan)
{
  int h, w, c;    
  //  NCHW
  unsigned char *argmax = (unsigned char *)malloc(sizeof(unsigned char) * width * height);
  for (h = 0; h < height; h++) {
    for (w = 0; w < width; w++) {
      float max = 0.0;
      argmax[h * width + w] = (unsigned char)0;
      for (c = 0; c < chan; c++) {
	int index = c * height * width + h * width + w;
	if (max < output[index]) {	  
	  max = output[index];
	  argmax[h * width + w] = (unsigned char)c;
	}
      }
    }
  }
  return argmax;
}

static image argmax2mask(unsigned char *argmax, int width, int height, int *colormap)
{
  image m = make_image(width, height, 3);
  int h, w;    
  for (h = 0; h < height; h++) {
    for (w = 0; w < width; w++) {
      int index = h * width + w;
      int value = argmax[index];
      m.data[w + width * h + width * height * 2] = colormap[value * 3 + 2]/255.0;
      m.data[w + width * h + width * height * 1] = colormap[value * 3 + 1]/255.0;
      m.data[w + width * h + width * height * 0] = colormap[value * 3 + 0]/255.0;
      /*
      if (argmax[index] == 0) { 
	m.data[w + width * h + width * height * 0] = 1.0;
      } else if (argmax[index] == 1) { 	
	m.data[w + width * h + width * height * 1] = 1.0;
      } else if (argmax[index] == 2) { 	
	m.data[w + width * h + width * height * 2] = 1.0;
      } else {
	m.data[w + width * h + width * height * 2] = 1.0;
	m.data[w + width * h + width * height * 1] = 1.0;
	m.data[w + width * h + width * height * 0] = 1.0;		
      }
      */
    }
  }
  return m;
}

void train_segmenter(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear, int dont_show, int calc_map, float thresh, float iou_thresh, int mjpeg_port, int show_imgs, int benchmark_layers, char* chart_path)
{
    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.txt");
    char *valid_images = option_find_str(options, "valid", train_images);
    char *backup_directory = option_find_str(options, "backup", "/backup/");

    network net_map;
    if (calc_map) {
        FILE* valid_file = fopen(valid_images, "r");
        if (!valid_file) {
            printf("\n Error: There is no %s file for mAP calculation!\n Don't use -map flag.\n Or set valid=%s in your %s file. \n", valid_images, train_images, datacfg);
            getchar();
            exit(-1);
        }
        else fclose(valid_file);

        cuda_set_device(gpus[0]);
        printf(" Prepare additional network for mAP calculation...\n");
        net_map = parse_network_cfg_custom(cfgfile, 1, 1);
        net_map.benchmark_layers = benchmark_layers;
        const int net_classes = net_map.layers[net_map.n - 1].classes;

        int k;  // free memory unnecessary arrays
        for (k = 0; k < net_map.n - 1; ++k) free_layer_custom(net_map.layers[k], 1);

        char *name_list = option_find_str(options, "names", "data/names.list");
        int names_size = 0;
        char **names = get_labels_custom(name_list, &names_size);
        if (net_classes != names_size) {
            printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
                name_list, names_size, net_classes, cfgfile);
            if (net_classes > names_size) getchar();
        }
        free_ptrs((void**)names, net_map.layers[net_map.n - 1].classes);
    }

    srand(time(0));
    char *base = basecfg(cfgfile);
    printf("%s\n", base);
    float avg_loss = -1;
    float avg_contrastive_acc = 0;
    network* nets = (network*)xcalloc(ngpus, sizeof(network));

    srand(time(0));
    int seed = rand();
    int k;
    for (k = 0; k < ngpus; ++k) {
        srand(seed);
#ifdef GPU
        cuda_set_device(gpus[k]);
#endif
        nets[k] = parse_network_cfg(cfgfile);
        nets[k].benchmark_layers = benchmark_layers;
        if (weightfile) {
            load_weights(&nets[k], weightfile);
	    sparsify_weights(&nets[k]);	    
        }
        if (clear) {
            *nets[k].seen = 0;
            *nets[k].cur_iteration = 0;
        }
        nets[k].learning_rate *= ngpus;
    }
    srand(time(0));
    network net = nets[0];

    const int actual_batch_size = net.batch * net.subdivisions;
    if (actual_batch_size == 1) {
        printf("\n Error: You set incorrect value batch=1 for Training! You should set batch=64 subdivision=64 \n");
        getchar();
    }
    else if (actual_batch_size < 8) {
        printf("\n Warning: You set batch=%d lower than 64! It is recommended to set batch=64 subdivision=64 \n", actual_batch_size);
    }

    int imgs = net.batch * net.subdivisions * ngpus;
    printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    data train, buffer;

    layer l = net.layers[net.n - 1];
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == SOFTMAX) {
            l = lk;
            printf(" Segmentation layer: %d - type = %d \n", k, l.type);
	    net.layers[net.n - 1].truths = l.w * l.h;
        }
    }

    int classes = l.classes;

    list *plist = get_paths(train_images);
    int train_images_num = plist->size;
    char **paths = (char **)list_to_array(plist);

    const int init_w = net.w;
    const int init_h = net.h;
    const int init_b = net.batch;
    int iter_save, iter_save_last, iter_map;
    iter_save = get_current_iteration(net);
    iter_save_last = get_current_iteration(net);
    iter_map = get_current_iteration(net);
    float mean_average_precision = -1;
    float best_map = mean_average_precision;

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.out_w = l.w;
    args.out_h = l.h;    
    args.paths = paths;
    args.n = imgs;
    args.m = plist->size;
    args.classes = classes;
    args.flip = net.flip;
    args.jitter = l.jitter;
    args.resize = l.resize;
    args.num_boxes = l.max_boxes;
    args.truth_size = l.w * l.h;
    net.num_boxes = args.num_boxes;
    net.train_images_num = train_images_num;
    args.d = &buffer;
    args.type = SEGMENTATION_DATA;
    args.threads = 64;    // 16 or 64

    args.angle = net.angle;
    args.gaussian_noise = net.gaussian_noise;
    args.blur = net.blur;
    args.mixup = net.mixup;
    args.exposure = net.exposure;
    args.saturation = net.saturation;
    args.hue = net.hue;
    args.letter_box = net.letter_box;
    args.mosaic_bound = net.mosaic_bound;
    args.contrastive = net.contrastive;
    args.contrastive_jit_flip = net.contrastive_jit_flip;
    args.contrastive_color = net.contrastive_color;
    if (dont_show && show_imgs) show_imgs = 2;
    args.show_imgs = show_imgs;

#ifdef OPENCV
    //int num_threads = get_num_threads();
    //if(num_threads > 2) args.threads = get_num_threads() - 2;
    args.threads = 6 * ngpus;   // 3 for - Amazon EC2 Tesla V100: p3.2xlarge (8 logical cores) - p3.16xlarge
    //args.threads = 12 * ngpus;    // Ryzen 7 2700X (16 logical cores)
    mat_cv* img = NULL;
    float max_img_loss = net.max_chart_loss;
    int number_of_lines = 100;
    int img_size = 1000;
    char windows_name[100];
    sprintf(windows_name, "chart_%s.png", base);
    img = draw_train_chart(windows_name, max_img_loss, net.max_batches, number_of_lines, img_size, dont_show, chart_path);
#endif    //OPENCV
    if (net.contrastive && args.threads > net.batch/2) args.threads = net.batch / 2;
    if (net.track) {
        args.track = net.track;
        args.augment_speed = net.augment_speed;
        if (net.sequential_subdivisions) args.threads = net.sequential_subdivisions * ngpus;
        else args.threads = net.subdivisions * ngpus;
        args.mini_batch = net.batch / net.time_steps;
        printf("\n Tracking! batch = %d, subdiv = %d, time_steps = %d, mini_batch = %d \n", net.batch, net.subdivisions, net.time_steps, args.mini_batch);
    }
    //printf(" imgs = %d \n", imgs);

    pthread_t load_thread = load_data(args);

    int count = 0;
    double time_remaining, avg_time = -1, alpha_time = 0.01;

    char db_name[4096];
    sprintf(db_name, "%s/%s.sqlite",backup_directory, base);
    sqlite3 *db = open_db(db_name);
    create_loss_table(db);
    create_eval_segmentation(db);
    //while(i*imgs < N*120){
    while (get_current_iteration(net) < net.max_batches) {
        if (l.random && count++ % 10 == 0) {
            float rand_coef = 1.4;
            if (l.random != 1.0) rand_coef = l.random;
            printf("Resizing, random_coef = %.2f \n", rand_coef);
            float random_val = rand_scale(rand_coef);    // *x or /x
            int dim_w = roundl(random_val*init_w / net.resize_step + 1) * net.resize_step;
            int dim_h = roundl(random_val*init_h / net.resize_step + 1) * net.resize_step;
            if (random_val < 1 && (dim_w > init_w || dim_h > init_h)) dim_w = init_w, dim_h = init_h;

            int max_dim_w = roundl(rand_coef*init_w / net.resize_step + 1) * net.resize_step;
            int max_dim_h = roundl(rand_coef*init_h / net.resize_step + 1) * net.resize_step;

            // at the beginning (check if enough memory) and at the end (calc rolling mean/variance)
            if (avg_loss < 0 || get_current_iteration(net) > net.max_batches - 100) {
                dim_w = max_dim_w;
                dim_h = max_dim_h;
            }

            if (dim_w < net.resize_step) dim_w = net.resize_step;
            if (dim_h < net.resize_step) dim_h = net.resize_step;
            int dim_b = (init_b * max_dim_w * max_dim_h) / (dim_w * dim_h);
            int new_dim_b = (int)(dim_b * 0.8);
            if (new_dim_b > init_b) dim_b = new_dim_b;

            args.w = dim_w;
            args.h = dim_h;

            int k;
            if (net.dynamic_minibatch) {
                for (k = 0; k < ngpus; ++k) {
                    (*nets[k].seen) = init_b * net.subdivisions * get_current_iteration(net); // remove this line, when you will save to weights-file both: seen & cur_iteration
                    nets[k].batch = dim_b;
                    int j;
                    for (j = 0; j < nets[k].n; ++j)
                        nets[k].layers[j].batch = dim_b;
                }
                net.batch = dim_b;
                imgs = net.batch * net.subdivisions * ngpus;
                args.n = imgs;
                printf("\n %d x %d  (batch = %d) \n", dim_w, dim_h, net.batch);
            }
            else
                printf("\n %d x %d \n", dim_w, dim_h);

            pthread_join(load_thread, 0);
            train = buffer;
            free_data(train);
            load_thread = load_data(args);

            for (k = 0; k < ngpus; ++k) {
                resize_network(nets + k, dim_w, dim_h);
            }
            net = nets[0];
        }
        double time = what_time_is_it_now();
        pthread_join(load_thread, 0);
        train = buffer;
        if (net.track) {
            net.sequential_subdivisions = get_current_seq_subdivisions(net);
            args.threads = net.sequential_subdivisions * ngpus;
            printf(" sequential_subdivisions = %d, sequence = %d \n", net.sequential_subdivisions, get_sequence_value(net));
        }
        load_thread = load_data(args);
        //wait_key_cv(500);

        /*
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        if(!b.x) break;
        printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
        }
        image im = float_to_image(448, 448, 3, train.X.vals[10]);
        int k;
        for(k = 0; k < l.max_boxes; ++k){
        box b = float_to_box(train.y.vals[10] + 1 + k*5);
        printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
        draw_bbox(im, b, 8, 1,0,0);
        }
        save_image(im, "truth11");
        */

        const double load_time = (what_time_is_it_now() - time);
        printf("Loaded: %lf seconds", load_time);
        if (load_time > 0.1 && avg_loss > 0) printf(" - performance bottleneck on CPU or Disk HDD/SSD");
        printf("\n");

        time = what_time_is_it_now();
        float loss = 0;
#ifdef GPU
        if (ngpus == 1) {
            int wait_key = (dont_show) ? 0 : 1;
            loss = train_network_waitkey(net, train, wait_key);
        }
        else {
            loss = train_networks(nets, ngpus, train, 4);
        }
#else
        loss = train_network(net, train);
#endif
        if (avg_loss < 0 || avg_loss != avg_loss) avg_loss = loss;    // if(-inf or nan)
        avg_loss = avg_loss*.9 + loss*.1;

        const int iteration = get_current_iteration(net);
        //i = get_current_batch(net);

	insert_loss(db, iteration, train_images_num, avg_loss, loss, get_current_rate(net), net.batch, ngpus);
	
        int calc_map_for_each = 4 * train_images_num / (net.batch * net.subdivisions);  // calculate mAP for each 4 Epochs
        calc_map_for_each = fmax(calc_map_for_each, 100);
        int next_map_calc = iter_map + calc_map_for_each;
        next_map_calc = fmax(next_map_calc, net.burn_in);
        //next_map_calc = fmax(next_map_calc, 400);
        if (calc_map) {
            printf("\n (next mAP calculation at %d iterations) ", next_map_calc);
            if (mean_average_precision > 0) printf("\n Last accuracy mAP@%0.2f = %2.2f %%, best = %2.2f %% ", iou_thresh, mean_average_precision * 100, best_map * 100);
        }

        #ifndef WIN32
        if (mean_average_precision > 0.0) {
            printf("\033]2;%d/%d: loss=%0.1f map=%0.2f best=%0.2f hours left=%0.1f\007", iteration, net.max_batches, loss, mean_average_precision, best_map, avg_time);
        }
        else {
            printf("\033]2;%d/%d: loss=%0.1f hours left=%0.1f\007", iteration, net.max_batches, loss, avg_time);
        }
        #endif

        if (net.cudnn_half) {
            if (iteration < net.burn_in * 3) fprintf(stderr, "\n Tensor Cores are disabled until the first %d iterations are reached.\n", 3 * net.burn_in);
            else fprintf(stderr, "\n Tensor Cores are used.\n");
            fflush(stderr);
        }
        printf("\n %d: %f, %f avg loss, %f rate, %lf seconds, %d images, %f hours left\n", iteration, loss, avg_loss, get_current_rate(net), (what_time_is_it_now() - time), iteration*imgs, avg_time);
        fflush(stdout);

        int draw_precision = 0;
        if (calc_map && (iteration >= next_map_calc || iteration == net.max_batches)) {
            if (l.random) {
                printf("Resizing to initial size: %d x %d ", init_w, init_h);
                args.w = init_w;
                args.h = init_h;
                int k;
                if (net.dynamic_minibatch) {
                    for (k = 0; k < ngpus; ++k) {
                        for (k = 0; k < ngpus; ++k) {
                            nets[k].batch = init_b;
                            int j;
                            for (j = 0; j < nets[k].n; ++j)
                                nets[k].layers[j].batch = init_b;
                        }
                    }
                    net.batch = init_b;
                    imgs = init_b * net.subdivisions * ngpus;
                    args.n = imgs;
                    printf("\n %d x %d  (batch = %d) \n", init_w, init_h, init_b);
                }
                pthread_join(load_thread, 0);
                free_data(train);
                train = buffer;
                load_thread = load_data(args);
                for (k = 0; k < ngpus; ++k) {
                    resize_network(nets + k, init_w, init_h);
                }
                net = nets[0];
            }

            copy_weights_net(net, &net_map);

            // combine Training and Validation networks
            //network net_combined = combine_train_valid_networks(net, net_map);

            iter_map = iteration;
            mean_average_precision = validate_segmenter_iou(datacfg, cfgfile, weightfile, thresh, iou_thresh, 0, net.letter_box, &net_map, db);// &net_combined);
            printf("\n mean_average_precision (mAP@%0.2f) = %f \n", iou_thresh, mean_average_precision);
            if (mean_average_precision >= best_map) {
                best_map = mean_average_precision;
                printf("New best mAP!\n");
                char buff[256];
                sprintf(buff, "%s/%s_best.weights", backup_directory, base);
                save_weights(net, buff);
            }

            draw_precision = 1;
        }
        time_remaining = ((net.max_batches - iteration) / ngpus)*(what_time_is_it_now() - time + load_time) / 60 / 60;
        // set initial value, even if resume training from 10000 iteration
        if (avg_time < 0) avg_time = time_remaining;
        else avg_time = alpha_time * time_remaining + (1 -  alpha_time) * avg_time;
#ifdef OPENCV
        if (net.contrastive) {
            float cur_con_acc = -1;
            for (k = 0; k < net.n; ++k)
                if (net.layers[k].type == CONTRASTIVE) cur_con_acc = *net.layers[k].loss;
            if (cur_con_acc >= 0) avg_contrastive_acc = avg_contrastive_acc*0.99 + cur_con_acc * 0.01;
            printf("  avg_contrastive_acc = %f \n", avg_contrastive_acc);
        }
        draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, net.max_batches, mean_average_precision, draw_precision, "mAP%", avg_contrastive_acc / 100, dont_show, mjpeg_port, avg_time);
#endif    // OPENCV

        //if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
        //if (i % 100 == 0) {
        if ((iteration >= (iter_save + 10000) || iteration % 10000 == 0) ||
            (iteration >= (iter_save + 1000) || iteration % 1000 == 0) && net.max_batches < 10000)
        {
            iter_save = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_%d.weights", backup_directory, base, iteration);
            save_weights(net, buff);
        }

        if (iteration >= (iter_save_last + 100) || (iteration % 100 == 0 && iteration > 1)) {
            iter_save_last = iteration;
#ifdef GPU
            if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
            char buff[256];
            sprintf(buff, "%s/%s_last.weights", backup_directory, base);
            save_weights(net, buff);

            if (net.ema_alpha && is_ema_initialized(net)) {
                sprintf(buff, "%s/%s_ema.weights", backup_directory, base);
                save_weights_upto(net, buff, net.n, 1);
                printf(" EMA weights are saved to the file: %s \n", buff);
            }
        }
        free_data(train);
    }
#ifdef GPU
    if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
    char buff[256];
    sprintf(buff, "%s/%s_final.weights", backup_directory, base);
    save_weights(net, buff);
    printf("If you want to train from the beginning, then use flag in the end of training command: -clear \n");

#ifdef OPENCV
    release_mat(&img);
    destroy_all_windows_cv();
#endif

    // free memory
    pthread_join(load_thread, 0);
    free_data(buffer);

    free_load_threads(&args);

    free(base);
    free(paths);
    free_list_contents(plist);
    free_list(plist);

    free_list_contents_kvp(options);
    free_list(options);

    for (k = 0; k < ngpus; ++k) free_network(nets[k]);
    free(nets);
    //free_network(net);

    if (calc_map) {
        net_map.n = 0;
        free_network(net_map);
    }
}



void print_segmenter_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

static void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int myclass = j;
            if (dets[i].prob[myclass] > 0) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[myclass],
                xmin, ymin, xmax, ymax);
        }
    }
}

static void print_kitti_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h, char *outfile, char *prefix)
{
    char *kitti_ids[] = { "car", "pedestrian", "cyclist" };
    FILE *fpd = 0;
    char buffd[1024];
    snprintf(buffd, 1024, "%s/%s/data/%s.txt", prefix, outfile, id);

    fpd = fopen(buffd, "w");
    int i, j;
    for (i = 0; i < total; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j)
        {
            //if (dets[i].prob[j]) fprintf(fpd, "%s 0 0 0 %f %f %f %f -1 -1 -1 -1 0 0 0 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
            if (dets[i].prob[j]) fprintf(fpd, "%s -1 -1 -10 %f %f %f %f -1 -1 -1 -1000 -1000 -1000 -10 %f\n", kitti_ids[j], xmin, ymin, xmax, ymax, dets[i].prob[j]);
        }
    }
    fclose(fpd);
}

static void eliminate_bdd(char *buf, char *a)
{
    int n = 0;
    int i, k;
    for (i = 0; buf[i] != '\0'; i++)
    {
        if (buf[i] == a[n])
        {
            k = i;
            while (buf[i] == a[n])
            {
                if (a[++n] == '\0')
                {
                    for (k; buf[k + n] != '\0'; k++)
                    {
                        buf[k] = buf[k + n];
                    }
                    buf[k] = '\0';
                    break;
                }
                i++;
            }
            n = 0; i--;
        }
    }
}

static void get_bdd_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    eliminate_bdd(p, ".jpg");
    eliminate_bdd(p, "/");
    strcpy(filename, p);
}

static void print_bdd_detections(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    char *bdd_ids[] = { "bike" , "bus" , "car" , "motor" ,"person", "rider", "traffic light", "traffic sign", "train", "truck" };
    get_bdd_image_id(image_path);
    int i, j;

    for (i = 0; i < num_boxes; ++i)
    {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx1 = xmin;
        float by1 = ymin;
        float bx2 = xmax;
        float by2 = ymax;

        for (j = 0; j < classes; ++j)
        {
            if (dets[i].prob[j])
            {
                fprintf(fp, "\t{\n\t\t\"name\":\"%s\",\n\t\t\"category\":\"%s\",\n\t\t\"bbox\":[%f, %f, %f, %f],\n\t\t\"score\":%f\n\t},\n", image_path, bdd_ids[j], bx1, by1, bx2, by2, dets[i].prob[j]);
            }
        }
    }
}

void validate_segmenter(char *datacfg, char *cfgfile, char *weightfile, char *outfile)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.list");
    char *name_list = option_find_str(options, "names", "data/names.list");
    char *prefix = option_find_str(options, "results", "results");
    char **names = get_labels(name_list);
    char *mapf = option_find_str(options, "map", 0);
    int *map = 0;
    if (mapf) map = read_map(mapf);

    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    fprintf(stderr, "Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
    srand(time(0));

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    layer l = net.layers[net.n - 1];
    int k;
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
            l = lk;
            printf(" Detection layer: %d - type = %d \n", k, l.type);
        }
    }
    int classes = l.classes;

    char buff[1024];
    char *type = option_find_str(options, "eval", "voc");
    FILE *fp = 0;
    FILE **fps = 0;
    int coco = 0;
    int imagenet = 0;
    int bdd = 0;
    int kitti = 0;

    if (0 == strcmp(type, "coco")) {
        if (!outfile) outfile = "coco_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        coco = 1;
    }
    else if (0 == strcmp(type, "bdd")) {
        if (!outfile) outfile = "bdd_results";
        snprintf(buff, 1024, "%s/%s.json", prefix, outfile);
        fp = fopen(buff, "w");
        fprintf(fp, "[\n");
        bdd = 1;
    }
    else if (0 == strcmp(type, "kitti")) {
        char buff2[1024];
        if (!outfile) outfile = "kitti_results";
        printf("%s\n", outfile);
        snprintf(buff, 1024, "%s/%s", prefix, outfile);
        int mkd = make_directory(buff, 0777);
        snprintf(buff2, 1024, "%s/%s/data", prefix, outfile);
        int mkd2 = make_directory(buff2, 0777);
        kitti = 1;
    }
    else if (0 == strcmp(type, "imagenet")) {
        if (!outfile) outfile = "imagenet-detection";
        snprintf(buff, 1024, "%s/%s.txt", prefix, outfile);
        fp = fopen(buff, "w");
        imagenet = 1;
        classes = 200;
    }
    else {
        if (!outfile) outfile = "comp4_det_test_";
        fps = (FILE**) xcalloc(classes, sizeof(FILE *));
        for (j = 0; j < classes; ++j) {
            snprintf(buff, 1024, "%s/%s%s.txt", prefix, outfile, names[j]);
            fps[j] = fopen(buff, "w");
        }
    }


    int m = plist->size;
    int i = 0;
    int t;

    float thresh = .001;
    float nms = .6;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    args.type = IMAGE_DATA;
    const int letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;

    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {
        fprintf(stderr, "%d\n", i);
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }
        for (t = 0; t < nthreads && i + t < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }
        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            char *path = paths[i + t - nthreads];
            char *id = basecfg(path);
            float *X = val_resized[t].data;
            network_predict(net, X);
            int w = val[t].w;
            int h = val[t].h;
            int nboxes = 0;
            detection *dets = get_network_boxes(&net, w, h, thresh, .5, map, 0, &nboxes, letter_box);
            if (nms) {
                if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
                else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
            }

            if (coco) {
                print_cocos(fp, path, dets, nboxes, classes, w, h);
            }
            else if (imagenet) {
                print_imagenet_detections(fp, i + t - nthreads + 1, dets, nboxes, classes, w, h);
            }
            else if (bdd) {
                print_bdd_detections(fp, path, dets, nboxes, classes, w, h);
            }
            else if (kitti) {
                print_kitti_detections(fps, id, dets, nboxes, classes, w, h, outfile, prefix);
            }
            else {
                print_segmenter_detections(fps, id, dets, nboxes, classes, w, h);
            }

            free_detections(dets, nboxes);
            free(id);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    if (fps) {
        for (j = 0; j < classes; ++j) {
            fclose(fps[j]);
        }
        free(fps);
    }
    if (coco) {
#ifdef WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
    }

    if (bdd) {
#ifdef WIN32
        fseek(fp, -3, SEEK_CUR);
#else
        fseek(fp, -2, SEEK_CUR);
#endif
        fprintf(fp, "\n]\n");
        fclose(fp);
    }

    if (fp) fclose(fp);

    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    fprintf(stderr, "Total Detection Time: %f Seconds\n", (double)time(0) - start);
}

void validate_segmenter_recall(char *datacfg, char *cfgfile, char *weightfile)
{
    network net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    srand(time(0));

    //list *plist = get_paths("data/coco_val_5k.list");
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    //layer l = net.layers[net.n - 1];

    int j, k;

    int m = plist->size;
    int i = 0;

    float thresh = .001;
    float iou_thresh = .5;
    float nms = .4;

    int total = 0;
    int correct = 0;
    int proposals = 0;
    float avg_iou = 0;

    for (i = 0; i < m; ++i) {
        char *path = paths[i];
        image orig = load_image(path, 0, 0, net.c);
        image sized = resize_image(orig, net.w, net.h);
        char *id = basecfg(path);
        network_predict(net, sized.data);
        int nboxes = 0;
        int letterbox = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, .5, 0, 1, &nboxes, letterbox);
        if (nms) do_nms_obj(dets, nboxes, 1, nms);

        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        for (k = 0; k < nboxes; ++k) {
            if (dets[k].objectness > thresh) {
                ++proposals;
            }
        }
        for (j = 0; j < num_labels; ++j) {
            ++total;
            box t = { truth[j].x, truth[j].y, truth[j].w, truth[j].h };
            float best_iou = 0;
            for (k = 0; k < nboxes; ++k) {
                float iou = box_iou(dets[k].bbox, t);
                if (dets[k].objectness > thresh && iou > best_iou) {
                    best_iou = iou;
                }
            }
            avg_iou += best_iou;
            if (best_iou > iou_thresh) {
                ++correct;
            }
        }
        //fprintf(stderr, " %s - %s - ", paths[i], labelpath);
        fprintf(stderr, "%5d %5d %5d\tRPs/Img: %.2f\tIOU: %.2f%%\tRecall:%.2f%%\n", i, correct, total, (float)proposals / (i + 1), avg_iou * 100 / total, 100.*correct / total);
        free(truth);
        free(id);
        free_image(orig);
        free_image(sized);
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

static int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(const box_prob *)pa;
    box_prob b = *(const box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

float validate_segmenter_iou(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net, sqlite3 *db)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "valid", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);
    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network net;
    char *base = basecfg(cfgfile);    
    char db_name[4096];
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_images = option_find_str(options, "train", "data/train.txt");
    list *plist_train = get_paths(train_images);
    int train_images_num = plist_train->size;        
    sprintf(db_name, "%s/%s.sqlite",backup_directory, base);
    int flg_open_db = false;
    if (!db) {
      flg_open_db = true;
      db = open_db(db_name);      
    }
    create_eval_segmentation(db);
    create_eval_segmentation_per_image(db);        
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
        remember_network_recurrent_state(*existing_net);
        free_network_recurrent_state(*existing_net);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }

    int iteration = get_current_iteration(net);

    int batch = net.batch;    
    /*
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        getchar();
    }
    */
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    list *plist_dif = NULL;
    char **paths_dif = NULL;
    if (difficult_valid_images) {
        plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }

    layer l = net.layers[net.n - 1];
    int k;
    int out_w = -1; 
    int out_h = -1;   
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == SOFTMAX) {
            l = lk;
            printf(" Segmentation layer: %d - type = %d \n", k, l.type);
	    net.layers[net.n - 1].truths = l.w * l.h;
	    out_w = l.w;
	    out_h = l.h;	    
	    break;
        }
    }
    int m = plist->size;
    int i = 0;
    int t, c;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;
    else args.type = IMAGE_DATA;


    int classes = l.inputs / l.groups;    
    Eval_segmentation *eval = (Eval_segmentation *)malloc(sizeof(Eval_segmentation) * classes);
    for (c = 0; c < classes; c++) {
      eval[c].tp = 0;
      eval[c].fn = 0;
      eval[c].fp = 0;            
    }
    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {

        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }

        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }

        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
	    printf("@@ %s %d,%d\n", path, i, t);
            float *X = val_resized[t].data;
            network_predict(net, X);
	    cuda_pull_array(l.output_gpu, l.output, l.w * l.h);
	
            char labelpath[4096];
	    replace_image_to_mask(path, labelpath);	    
	    mat_cv *mask = load_image_mat_cv(labelpath, 0);
	    if (mask == NULL) {
	      printf("\n Error in load_mask_image() - OpenCV \n");	      
	    }
            int oh = get_height_mat(mask);
            int ow = get_width_mat(mask);	    
            image am = mask_data_augmentation(mask, out_w, out_h, 0, 0, ow, oh, 0, 0.0, 1.0, 1.0,
                0, 0, 0, 0, NULL);
	    //	    save_image_png(am, "mask");
	    unsigned char *argmax = apply_argmax(l.output, l.w, l.h, l.c);
	    Eval_segmentation *ret = evaluate_segmentation(argmax, am.data, out_w, out_h, classes);
	    show_metrics(ret, classes, labelpath, names);
	    for (c = 0; c < classes; c++) {
	      eval[c].tp += ret[c].tp;
	      eval[c].fn += ret[c].fn;
	      eval[c].fp += ret[c].fp;
	      insert_eval_segmentation_per_image(db, ret[c], "val", names[c], iteration, train_images_num, batch, 1, labelpath);	      	      
	    }
	    free(ret);
	    free(argmax);
	    free_image(am);
            release_mat(&mask);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    char mean_metrics[4096] = "\n\nMean Metrics";
    show_metrics(eval, classes, mean_metrics, names);
    for (c = 0; c < classes; c++) {
      insert_eval_segmentation(db, eval[c], "val", names[c], iteration, train_images_num, batch, 1);
    }
    free(eval);

    free(base);
    free_list_contents(plist_train);
    free_list(plist_train);    
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    
    // free memory
    free_ptrs((void**)names, names_size);
    free_list_contents_kvp(options);
    free_list(options);

    
    if (existing_net) {
        //set_batch_network(&net, initial_batch);
        //free_network_recurrent_state(*existing_net);
        restore_network_recurrent_state(*existing_net);
        //randomize_network_recurrent_state(*existing_net);
    }
    else {
        free_network(net);
    }
    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    if (flg_open_db) {
      close_db(db);
    }
    return 0;
}


float uncertain_segmenter_iou(char *datacfg, char *cfgfile, char *weightfile, float thresh_calc_avg_iou, const float iou_thresh, const int map_points, int letter_box, network *existing_net, sqlite3 *db)
{
    int j;
    list *options = read_data_cfg(datacfg);
    char *valid_images = option_find_str(options, "train", "data/train.txt");
    char *difficult_valid_images = option_find_str(options, "difficult", NULL);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);
    //char *mapf = option_find_str(options, "map", 0);
    //int *map = 0;
    //if (mapf) map = read_map(mapf);
    FILE* reinforcement_fd = NULL;

    network net;
    char *base = basecfg(cfgfile);    
    char db_name[4096];
    char *backup_directory = option_find_str(options, "backup", "/backup/");
    char *train_images = option_find_str(options, "train", "data/train.txt");
    list *plist_train = get_paths(train_images);
    int train_images_num = plist_train->size;        
    sprintf(db_name, "%s/%s.sqlite",backup_directory, base);
    int flg_open_db = false;
    if (!db) {
      flg_open_db = true;
      db = open_db(db_name);      
    }
    create_eval_segmentation(db);
    create_eval_segmentation_per_image(db);        
    //int initial_batch;
    if (existing_net) {
        char *train_images = option_find_str(options, "train", "data/train.txt");
        valid_images = option_find_str(options, "valid", train_images);
        net = *existing_net;
        remember_network_recurrent_state(*existing_net);
        free_network_recurrent_state(*existing_net);
    }
    else {
        net = parse_network_cfg_custom(cfgfile, 1, 1);    // set batch=1
        if (weightfile) {
            load_weights(&net, weightfile);
        }
        //set_batch_network(&net, 1);
        fuse_conv_batchnorm(net);
        calculate_binary_weights(net);
    }

    int iteration = get_current_iteration(net);

    int batch = net.batch;    
    /*
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        getchar();
    }
    */
    srand(time(0));
    printf("\n calculation mAP (mean average precision)...\n");

    list *plist = get_paths(valid_images);
    char **paths = (char **)list_to_array(plist);

    list *plist_dif = NULL;
    char **paths_dif = NULL;
    if (difficult_valid_images) {
        plist_dif = get_paths(difficult_valid_images);
        paths_dif = (char **)list_to_array(plist_dif);
    }

    layer l = net.layers[net.n - 1];
    int k;
    int out_w = -1; 
    int out_h = -1;   
    for (k = 0; k < net.n; ++k) {
        layer lk = net.layers[k];
        if (lk.type == SOFTMAX) {
            l = lk;
            printf(" Segmentation layer: %d - type = %d \n", k, l.type);
	    net.layers[net.n - 1].truths = l.w * l.h;
	    out_w = l.w;
	    out_h = l.h;	    
	    break;
        }
    }
    int m = plist->size;
    int i = 0;
    int t, c;

    int nthreads = 4;
    if (m < 4) nthreads = m;
    image* val = (image*)xcalloc(nthreads, sizeof(image));
    image* val_resized = (image*)xcalloc(nthreads, sizeof(image));
    image* buf = (image*)xcalloc(nthreads, sizeof(image));
    image* buf_resized = (image*)xcalloc(nthreads, sizeof(image));
    pthread_t* thr = (pthread_t*)xcalloc(nthreads, sizeof(pthread_t));

    load_args args = { 0 };
    args.w = net.w;
    args.h = net.h;
    args.c = net.c;
    letter_box = net.letter_box;
    if (letter_box) args.type = LETTERBOX_DATA;
    else args.type = IMAGE_DATA;


    int classes = l.inputs / l.groups;    
    Eval_segmentation *eval = (Eval_segmentation *)malloc(sizeof(Eval_segmentation) * classes);
    for (c = 0; c < classes; c++) {
      eval[c].tp = 0;
      eval[c].fn = 0;
      eval[c].fp = 0;            
    }
    for (t = 0; t < nthreads; ++t) {
        args.path = paths[i + t];
        args.im = &buf[t];
        args.resized = &buf_resized[t];
        thr[t] = load_data_in_thread(args);
    }
    time_t start = time(0);
    for (i = nthreads; i < m + nthreads; i += nthreads) {

        fprintf(stderr, "\r%d", i);
        for (t = 0; t < nthreads && (i + t - nthreads) < m; ++t) {
            pthread_join(thr[t], 0);
            val[t] = buf[t];
            val_resized[t] = buf_resized[t];
        }

        for (t = 0; t < nthreads && (i + t) < m; ++t) {
            args.path = paths[i + t];
            args.im = &buf[t];
            args.resized = &buf_resized[t];
            thr[t] = load_data_in_thread(args);
        }

        for (t = 0; t < nthreads && i + t - nthreads < m; ++t) {
            const int image_index = i + t - nthreads;
            char *path = paths[image_index];
	    printf("@@ %s %d,%d\n", path, i, t);
            float *X = val_resized[t].data;
            network_predict(net, X);
	    cuda_pull_array(l.output_gpu, l.output, l.w * l.h);
	
            char labelpath[4096];
	    replace_image_to_mask(path, labelpath);	    
	    mat_cv *mask = load_image_mat_cv(labelpath, 0);
	    if (mask == NULL) {
	      printf("\n Error in load_mask_image() - OpenCV \n");	      
	    }
            int oh = get_height_mat(mask);
            int ow = get_width_mat(mask);	    
            image am = mask_data_augmentation(mask, out_w, out_h, 0, 0, ow, oh, 0, 0.0, 1.0, 1.0,
                0, 0, 0, 0, NULL);
	    //	    save_image_png(am, "mask");
	    unsigned char *argmax = apply_argmax(l.output, l.w, l.h, l.c);
	    Eval_segmentation *ret = evaluate_segmentation(argmax, am.data, out_w, out_h, classes);
	    show_metrics(ret, classes, labelpath, names);
	    for (c = 0; c < classes; c++) {
	      eval[c].tp += ret[c].tp;
	      eval[c].fn += ret[c].fn;
	      eval[c].fp += ret[c].fp;
	      insert_eval_segmentation_per_image(db, ret[c], "train", names[c], iteration, train_images_num, batch, 1, labelpath);	      	      
	    }
	    free(ret);
	    free(argmax);
	    free_image(am);
            release_mat(&mask);
            free_image(val[t]);
            free_image(val_resized[t]);
        }
    }
    char mean_metrics[4096] = "\n\nMean Metrics";
    show_metrics(eval, classes, mean_metrics, names);
    for (c = 0; c < classes; c++) {
      insert_eval_segmentation(db, eval[c], "train", names[c], iteration, train_images_num, batch, 1);
    }
    free(eval);

    free(base);
    free_list_contents(plist_train);
    free_list(plist_train);    
    free_list_contents(plist);
    free_list(plist);
    free(paths);
    
    // free memory
    free_ptrs((void**)names, names_size);
    free_list_contents_kvp(options);
    free_list(options);

    
    if (existing_net) {
        //set_batch_network(&net, initial_batch);
        //free_network_recurrent_state(*existing_net);
        restore_network_recurrent_state(*existing_net);
        //randomize_network_recurrent_state(*existing_net);
    }
    else {
        free_network(net);
    }
    if (val) free(val);
    if (val_resized) free(val_resized);
    if (thr) free(thr);
    if (buf) free(buf);
    if (buf_resized) free(buf_resized);

    if (flg_open_db) {
      close_db(db);
    }
    return 0;
}

typedef struct {
    float w, h;
} anchors_t;

static int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(const anchors_t *)pa;
    anchors_t b = *(const anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

static int anchors_data_comparator(const float **pa, const float **pb)
{
    float *a = (float *)*pa;
    float *b = (float *)*pb;
    float diff = b[0] * b[1] - a[0] * a[1];
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}


static void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet segmenter calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }

    //float pointsdata[] = { 1,1, 2,2, 6,6, 5,5, 10,10 };
    float* rel_width_height_array = (float*)xcalloc(1000, sizeof(float));


    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    int classes = option_find_int(options, "classes", 1);
    int* counter_per_class = (int*)xcalloc(classes, sizeof(int));

    srand(time(0));
    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        //printf(" new path: %s \n", labelpath);
        char *buff = (char*)xcalloc(6144, sizeof(char));
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
                if (check_mistakes) getchar();
            }
            if (truth[j].id >= classes) {
                classes = truth[j].id + 1;
                counter_per_class = (int*)xrealloc(counter_per_class, classes * sizeof(int));
            }
            counter_per_class[truth[j].id]++;

            number_of_boxes++;
            rel_width_height_array = (float*)xrealloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));

            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i + 1, number_of_boxes);
        }
        free(buff);
        free(truth);
    }
    printf("\n all loaded. \n");
    printf("\n calculating k-means++ ...");

    matrix boxes_data;
    model anchors_data;
    boxes_data = make_matrix(number_of_boxes, 2);

    printf("\n");
    for (i = 0; i < number_of_boxes; ++i) {
        boxes_data.vals[i][0] = rel_width_height_array[i * 2];
        boxes_data.vals[i][1] = rel_width_height_array[i * 2 + 1];
        //if (w > 410 || h > 410) printf("i:%d,  w = %f, h = %f \n", i, w, h);
    }

    // Is used: distance(box, centroid) = 1 - IoU(box, centroid)

    // K-means
    anchors_data = do_kmeans(boxes_data, num_of_clusters);

    qsort((void*)anchors_data.centers.vals, num_of_clusters, 2 * sizeof(float), (__compar_fn_t)anchors_data_comparator);

    //gen_anchors.py = 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66
    //float orig_anch[] = { 1.19, 1.99, 2.79, 4.60, 4.53, 8.92, 8.06, 5.29, 10.32, 10.66 };

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = rel_width_height_array[i * 2]; //points->data.fl[i * 2];
        float box_h = rel_width_height_array[i * 2 + 1]; //points->data.fl[i * 2 + 1];
                                                         //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        float best_iou = 0;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = anchors_data.centers.vals[j][0];   // centers->data.fl[j * 2];
            float anchor_h = anchors_data.centers.vals[j][1];   // centers->data.fl[j * 2 + 1];
            float min_w = (box_w < anchor_w) ? box_w : anchor_w;
            float min_h = (box_h < anchor_h) ? box_h : anchor_h;
            float box_intersect = min_w*min_h;
            float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
            float iou = box_intersect / box_union;
            float distance = 1 - iou;
            if (distance < min_dist) {
              min_dist = distance;
              cluster_idx = j;
              best_iou = iou;
            }
        }

        float anchor_w = anchors_data.centers.vals[cluster_idx][0]; //centers->data.fl[cluster_idx * 2];
        float anchor_h = anchors_data.centers.vals[cluster_idx][1]; //centers->data.fl[cluster_idx * 2 + 1];
        if (best_iou > 1 || best_iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %f, box_h = %f, anchor_w = %f, anchor_h = %f, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, best_iou);
        }
        else avg_iou += best_iou;
    }

    char buff[1024];
    FILE* fwc = fopen("counters_per_class.txt", "wb");
    if (fwc) {
        sprintf(buff, "counters_per_class = ");
        printf("\n%s", buff);
        fwrite(buff, sizeof(char), strlen(buff), fwc);
        for (i = 0; i < classes; ++i) {
            sprintf(buff, "%d", counter_per_class[i]);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fwc);
            if (i < classes - 1) {
                fwrite(", ", sizeof(char), 2, fwc);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fwc);
    }
    else {
        printf(" Error: file counters_per_class.txt can't be open \n");
    }

    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);


    FILE* fw = fopen("anchors.txt", "wb");
    if (fw) {
        printf("\nSaving anchors to the file: anchors.txt \n");
        printf("anchors = ");
        for (i = 0; i < num_of_clusters; ++i) {
            float anchor_w = anchors_data.centers.vals[i][0]; //centers->data.fl[i * 2];
            float anchor_h = anchors_data.centers.vals[i][1]; //centers->data.fl[i * 2 + 1];
            if (width > 32) sprintf(buff, "%3.0f,%3.0f", anchor_w, anchor_h);
            else sprintf(buff, "%2.4f,%2.4f", anchor_w, anchor_h);
            printf("%s", buff);
            fwrite(buff, sizeof(char), strlen(buff), fw);
            if (i + 1 < num_of_clusters) {
                fwrite(", ", sizeof(char), 2, fw);
                printf(", ");
            }
        }
        printf("\n");
        fclose(fw);
    }
    else {
        printf(" Error: file anchors.txt can't be open \n");
    }

    if (show) {
#ifdef OPENCV
        show_acnhors(number_of_boxes, num_of_clusters, rel_width_height_array, anchors_data, width, height);
#endif // OPENCV
    }
    free(rel_width_height_array);
    free(counter_per_class);

    getchar();
}


void test_segmenter(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
            error("fopen failed", DARKNET_LOC);
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
	    if (lk.type == SOFTMAX) {	    
                l = lk;
                printf(" Segmentation layer: %d - type = %d \n", k, l.type);
            }
        }
        layer ld = net.layers[net.n - 1];	
	for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                ld = lk;
                printf(" Detection layer: %d - type = %d \n", k, ld.type);
            }
        }
	
        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);

        for (k = net.n-1; k >= 0; k--) {
            layer lk = net.layers[k];
	    if (lk.type == SOFTMAX) {	    
                l = lk;
                printf(" Segmentation layer: %d - type = %d \n", k, l.type);
		cuda_pull_array(l.output_gpu, l.output, l.w * l.h * l.c);
		unsigned char *argmax = apply_argmax(l.output, l.w, l.h, l.c);
		image mask = argmax2mask(argmax, l.w, l.h, l.map);
		save_image_png(mask, "mask");
		image resized_mask = resize_image(mask, im.w, im.h); //Attention : Bilinear Interpolation
		blend_images_cv(im, 1.0, resized_mask, 0.5);
		if (!dont_show) {
		  char buff[256];
		  sprintf(buff, "mask%d", k);		
		  show_image(mask, buff);	    
		}
		free(argmax);		
		free_image(mask);
		free_image(resized_mask);		
            }
        }	

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (ld.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, ld.classes, nms);
            else diounms_sort(dets, nboxes, ld.classes, nms, ld.nms_kind, ld.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, ld.classes, ext_output);
	
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        save_image(im, "predictions");

	if (!dont_show) {
	  show_image(im, "predictions");
	}
	
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }


    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}


void depth_segmenter(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
    float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    if (net.letter_box) letter_box = 1;
    net.benchmark_layers = benchmark_layers;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    char buff[256];
    char *input = buff;
    char *json_buf = NULL;
    int json_image_id = 0;
    FILE* json_file = NULL;
    if (outfile) {
        json_file = fopen(outfile, "wb");
        if(!json_file) {
            error("fopen failed", DARKNET_LOC);
        }
        char *tmp = "[\n";
        fwrite(tmp, sizeof(char), strlen(tmp), json_file);
    }
    int j, i;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if(letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
	    if (lk.type == SOFTMAX) {	    
                l = lk;
                printf(" Segmentation layer: %d - type = %d \n", k, l.type);
            }
        }
        layer ld = net.layers[net.n - 1];	
	for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                ld = lk;
                printf(" Detection layer: %d - type = %d \n", k, ld.type);
            }
        }
	
        //box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        //float **probs = calloc(l.w*l.h*l.n, sizeof(float*));
        //for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)xcalloc(l.classes, sizeof(float));

        float *X = sized.data;

        //time= what_time_is_it_now();
        double time = get_time_point();
        network_predict(net, X);

        for (k = net.n-1; k >= 0; k--) {
            layer lk = net.layers[k];
	    if (lk.type == SOFTMAX) {	    
                l = lk;
                printf(" Segmentation layer: %d - type = %d \n", k, l.type);
		cuda_pull_array(l.output_gpu, l.output, l.w * l.h * l.c);
		unsigned char *argmax = apply_argmax(l.output, l.w, l.h, l.c);
		image depth = make_image(l.w, l.h, 1);
		for(j = 0; j < l.h; ++j){
		  for(i = 0; i < l.w; ++i){
		    depth.data[i+l.w*j] = argmax[i+l.w*j]/(float)95;
		  }
		}		
		//image mask = argmax2mask(argmax, l.w, l.h, l.map);
		save_image_png(depth, "depth");

		if (!dont_show) {
		  char buff[256];
		  sprintf(buff, "mask%d", k);		
		  show_image(depth, buff);	    
		}
		free(argmax);		
		free_image(depth);
            }
        }	

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (ld.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, ld.classes, nms);
            else diounms_sort(dets, nboxes, ld.classes, nms, ld.nms_kind, ld.beta_nms);
        }
        draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, ld.classes, ext_output);
	
        printf("%s: Predicted in %lf milli-seconds.\n", input, ((double)get_time_point() - time) / 1000);
        save_image(im, "predictions");

	if (!dont_show) {
	  show_image(im, "predictions");
	}
	
        free_image(im);
        free_image(sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        if (filename) break;
    }


    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);


    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}

void merge_segmenter(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
		     float hier_thresh, int dont_show, int ext_output, int save_labels, char *outfile, int letter_box, int benchmark_layers, char *cfgfile2, char *weightfile2, int from)
{
    gpu_index = -1;
    network net = parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net, weightfile);
    }

    network net2 = parse_network_cfg_custom(cfgfile2, 1, 1); // set batch=1
    if (weightfile) {
        load_weights(&net2, weightfile2);
    }
    
    save_merge_weights_from_index(net, net2, "merge.weights", from, 0);
    free_network(net);
    free_network(net2);    
}

#if defined(OPENCV) && defined(GPU)

// adversarial attack dnn
static void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg(cfgfile);// parse_network_cfg_custom(cfgfile, 1, 1); // set batch=1
    net.adversarial = 1;
    set_batch_network(&net, 1);
    if (weightfile) {
        load_weights(&net, weightfile);
    }
    net.benchmark_layers = benchmark_layers;
    //fuse_conv_batchnorm(net);
    //calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if (net.layers[net.n - 1].classes > names_size) getchar();
    }

    srand(2222222);
    char buff[256];
    char *input = buff;

    int j;
    float nms = .45;    // 0.4F
    while (1) {
        if (filename) {
            strncpy(input, filename, 256);
            if (strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        }
        else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if (!input) break;
            strtok(input, "\n");
        }
        //image im;
        //image sized = load_image_resize(input, net.w, net.h, net.c, &im);
        image im = load_image(input, 0, 0, net.c);
        image sized;
        if (letter_box) sized = letterbox_image(im, net.w, net.h);
        else sized = resize_image(im, net.w, net.h);

        image src_sized = copy_image(sized);

        layer l = net.layers[net.n - 1];
        int k;
        for (k = 0; k < net.n; ++k) {
            layer lk = net.layers[k];
            if (lk.type == YOLO || lk.type == GAUSSIAN_YOLO || lk.type == REGION) {
                l = lk;
                printf(" Detection layer: %d - type = %d \n", k, l.type);
            }
        }

        net.num_boxes = l.max_boxes;
        int num_truth = l.truths;
        float *truth_cpu = (float *)xcalloc(num_truth, sizeof(float));

        int *it_num_set = (int *)xcalloc(1, sizeof(int));
        float *lr_set = (float *)xcalloc(1, sizeof(float));
        int *boxonly = (int *)xcalloc(1, sizeof(int));

        cv_draw_object(sized, truth_cpu, net.num_boxes, num_truth, it_num_set, lr_set, boxonly, l.classes, names);

        net.learning_rate = *lr_set;
        it_num = *it_num_set;

        float *X = sized.data;

        mat_cv* img = NULL;
        float max_img_loss = 5;
        int number_of_lines = 100;
        int img_size = 1000;
        char windows_name[100];
        char *base = basecfg(cfgfile);
        sprintf(windows_name, "chart_%s.png", base);
        img = draw_train_chart(windows_name, max_img_loss, it_num, number_of_lines, img_size, dont_show, NULL);

        int iteration;
        for (iteration = 0; iteration < it_num; ++iteration)
        {
            forward_backward_network_gpu(net, X, truth_cpu);

            float avg_loss = get_network_cost(net);
            draw_train_loss(windows_name, img, img_size, avg_loss, max_img_loss, iteration, it_num, 0, 0, "mAP%", 0, dont_show, 0, 0);

            float inv_loss = 1.0 / max_val_cmp(0.01, avg_loss);
            //net.learning_rate = *lr_set * inv_loss;

            if (*boxonly) {
                int dw = truth_cpu[2] * sized.w, dh = truth_cpu[3] * sized.h;
                int dx = truth_cpu[0] * sized.w - dw / 2, dy = truth_cpu[1] * sized.h - dh / 2;
                image crop = crop_image(sized, dx, dy, dw, dh);
                copy_image_inplace(src_sized, sized);
                embed_image(crop, sized, dx, dy);
            }

            show_image_cv(sized, "image_optimization");
            wait_key_cv(20);
        }

        net.train = 0;
        quantize_image(sized);
        network_predict(net, X);

        save_image_png(sized, "drawn");
        //sized = load_image("drawn.png", 0, 0, net.c);

        int nboxes = 0;
        detection *dets = get_network_boxes(&net, sized.w, sized.h, thresh, 0, 0, 1, &nboxes, letter_box);
        if (nms) {
            if (l.nms_kind == DEFAULT_NMS) do_nms_sort(dets, nboxes, l.classes, nms);
            else diounms_sort(dets, nboxes, l.classes, nms, l.nms_kind, l.beta_nms);
        }
        draw_detections_v3(sized, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        save_image(sized, "pre_predictions");
        if (!dont_show) {
            show_image(sized, "pre_predictions");
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);
        free_image(src_sized);

        if (!dont_show) {
            wait_until_press_key_cv();
            destroy_all_windows_cv();
        }

        free(lr_set);
        free(it_num_set);

        if (filename) break;
    }

    // free memory
    free_ptrs((void**)names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(net);
}
#else // defined(OPENCV) && defined(GPU)
void draw_object(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int dont_show, int it_num,
    int letter_box, int benchmark_layers)
{
    printf(" ./darknet segmenter draw ... can't be used without OpenCV and CUDA! \n");
    getchar();
}
#endif // defined(OPENCV) && defined(GPU)

void run_segmenter(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int benchmark = find_arg(argc, argv, "-benchmark");
    int benchmark_layers = find_arg(argc, argv, "-benchmark_layers");
    //if (benchmark_layers) benchmark = 1;
    if (benchmark) dont_show = 1;
    int show = find_arg(argc, argv, "-show");
    int letter_box = find_arg(argc, argv, "-letter_box");
    int calc_map = find_arg(argc, argv, "-map");
    int map_points = find_int_arg(argc, argv, "-points", 0);
    check_mistakes = find_arg(argc, argv, "-check_mistakes");
    int show_imgs = find_arg(argc, argv, "-show_imgs");
    int mjpeg_port = find_int_arg(argc, argv, "-mjpeg_port", -1);
    int avgframes = find_int_arg(argc, argv, "-avgframes", 3);
    int dontdraw_bbox = find_arg(argc, argv, "-dontdraw_bbox");
    int json_port = find_int_arg(argc, argv, "-json_port", -1);
    char *http_post_host = find_char_arg(argc, argv, "-http_post_host", 0);
    int time_limit_sec = find_int_arg(argc, argv, "-time_limit_sec", 0);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float iou_thresh = find_float_arg(argc, argv, "-iou_thresh", .5);    // 0.5 for mAP
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);
    // extended output in test mode (output of rect bound coords)
    // and for recall mode (extended output table-like format with results for best_class fit)
    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
    char* chart_path = find_char_arg(argc, argv, "-chart", 0);

    char *merge_cfg = find_char_arg(argc, argv, "-cfg", 0);
    char *merge_weights = find_char_arg(argc, argv, "-weights", 0);
    int from = find_int_arg(argc, argv, "-from", 0);    
    if (argc < 4) {
        fprintf(stderr, "usage: %s %s [train/test/valid/demo/map] [data] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if (gpu_list) {
        printf("%s\n", gpu_list);
        int len = (int)strlen(gpu_list);
        ngpus = 1;
        int i;
        for (i = 0; i < len; ++i) {
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = (int*)xcalloc(ngpus, sizeof(int));
        for (i = 0; i < ngpus; ++i) {
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',') + 1;
        }
    }
    else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }

    int clear = find_arg(argc, argv, "-clear");

    char *datacfg = argv[3];
    char *cfg = argv[4];
    char *weights = (argc > 5) ? argv[5] : 0;
    if (weights)
        if (strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = (argc > 6) ? argv[6] : 0;
    if (0 == strcmp(argv[2], "test")) test_segmenter(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);
    if (0 == strcmp(argv[2], "depth")) depth_segmenter(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers);    
    else if (0 == strcmp(argv[2], "train")) train_segmenter(datacfg, cfg, weights, gpus, ngpus, clear, dont_show, calc_map, thresh, iou_thresh, mjpeg_port, show_imgs, benchmark_layers, chart_path);
    else if (0 == strcmp(argv[2], "valid")) validate_segmenter(datacfg, cfg, weights, outfile);
    else if (0 == strcmp(argv[2], "recall")) validate_segmenter_recall(datacfg, cfg, weights);
    else if (0 == strcmp(argv[2], "iou")) validate_segmenter_iou(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL, NULL);
    else if (0 == strcmp(argv[2], "uncertain")) uncertain_segmenter_iou(datacfg, cfg, weights, thresh, iou_thresh, map_points, letter_box, NULL, NULL);    
    else if (0 == strcmp(argv[2], "merge")) merge_segmenter(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, outfile, letter_box, benchmark_layers, merge_cfg, merge_weights, from);
    else if (0 == strcmp(argv[2], "demo")) {
        list *options = read_data_cfg(datacfg);
        int classes = option_find_int(options, "classes", 20);
        char *name_list = option_find_str(options, "names", "data/names.list");
        char **names = get_labels(name_list);
        if (filename)
            if (strlen(filename) > 0)
                if (filename[strlen(filename) - 1] == 0x0d) filename[strlen(filename) - 1] = 0;
        demo(cfg, weights, thresh, hier_thresh, cam_index, filename, names, classes, avgframes, frame_skip, prefix, out_filename,
            mjpeg_port, dontdraw_bbox, json_port, dont_show, ext_output, letter_box, time_limit_sec, http_post_host, benchmark, benchmark_layers);

        free_list_contents_kvp(options);
        free_list(options);
    }
    else printf(" There isn't such command: %s", argv[2]);

    if (gpus && gpu_list && ngpus > 1) free(gpus);
}
