#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sqlite3.h>
#include <time.h>

typedef struct {
  float ap;
  float iou;  
  int tp;    
  int fn;
  int fp;
} Eval_detection;

typedef struct {
  int tp;    
  int fn;
  int fp;
} Eval_segmentation;

extern sqlite3 *open_db(const char *name);
extern void create_loss_table(sqlite3 *db);
extern void close_db(sqlite3 *db);
extern void insert_loss(sqlite3 *db, int iteration, int images, float avg_loss, float loss, double learning_rate, int batch, int gpus);
extern void create_eval_detection(sqlite3 *db);
extern void insert_eval_detection(sqlite3 *db, Eval_detection eval, char *mode, char *name, int iteration, int images, int batch, int gpus);  
extern void create_eval_segmentation(sqlite3 *db);
extern void insert_eval_segmentation(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus);
extern void create_eval_segmentation_per_image(sqlite3 *db);
extern void insert_eval_segmentation_per_image(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus, char *image_name);
extern void create_eval_detection_per_image(sqlite3 *db);
extern void insert_eval_detection_per_image(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus, char *image_name);
