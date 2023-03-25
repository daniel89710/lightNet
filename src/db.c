#include "db.h"

#define LENGTH 8192

sqlite3 *open_db(const char *name)
{
  sqlite3 *db=NULL;
  int ret = sqlite3_open(name, &db);
  if (SQLITE_OK !=ret){
    printf("Fail to open %s\n", name);
  }
  return db;
}

void close_db(sqlite3 *db)
{
  int ret = sqlite3_close(db);
  if( SQLITE_OK != ret ){
    printf("Fail to close db\n");
  }   
}

void create_loss_table(sqlite3 *db)
{
  char *err=NULL;
  int ret = sqlite3_exec(db, "create table loss (iteration integer, epoch integer, images integer, avg_loss float, loss float, learning_rate double, batch_size integer, gpus integer, year integer, month integer, day intger, hour integer, min integer, sec integer);", NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }  
}

void insert_loss(sqlite3 *db, int iteration, int images, float avg_loss, float loss, double learning_rate, int batch, int gpus)
{
  char *err=NULL;
  char sql[LENGTH];
  int year, month, day, hour, min, sec;
  time_t timer;
  struct tm *local;  
  int epoch = iteration/(int)(images/(batch/gpus)-0.5);
  timer = time(NULL);
  local = localtime(&timer);
  year = local->tm_year + 1900;
  month = local->tm_mon + 1;
  day = local->tm_mday;
  hour = local->tm_hour;
  min = local->tm_min;
  sec = local->tm_sec;  
  sprintf(sql, "INSERT INTO loss (iteration, epoch, images, avg_loss, loss, learning_rate , batch_size, gpus, year, month, day, hour, min, sec) values(%d, %d, %d, %f, %f, %.12f, %d, %d, %d,%d,%d,%d,%d,%d);",
	 iteration,
	 epoch,
	 images,	
	 avg_loss,
	 loss,
	 learning_rate,
	 batch,
	 gpus,
	 year,
	 month,
	 day,
	 hour,
	 min,
	 sec);
  //  printf("%s\n",sql);
  int ret = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }   
}

void create_eval_detection(sqlite3 *db)
{
  int ret = sqlite3_exec(db,  "CREATE TABLE eval (iteration integer, epoch integer, mode string, name string, ap float, iou float, tp integer, fn integer, fp integer);", NULL, NULL, NULL);
}


void insert_eval_detection(sqlite3 *db, Eval_detection eval, char *mode, char *name, int iteration, int images, int batch, int gpus)
{
  char *err=NULL;
  char sql[LENGTH];  
  int epoch = iteration/(int)(images/(batch/gpus)-0.5);
  sprintf(sql, "INSERT INTO eval (iteration, epoch, mode, name, ap, iou, tp, fn, fp) values(%d, %d, '%s', '%s', %f, %f, %d, %d, %d);",
	  iteration,
	  epoch,
	  mode,
	  name,
	  eval.ap,
	  eval.iou,
	  eval.tp,
	  eval.fn,
	  eval.fp);
  //  printf("%s\n",sql);
  int ret = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }   	  
}


void create_eval_segmentation(sqlite3 *db)
{
  int ret = sqlite3_exec(db,  "CREATE TABLE eval_seg (iteration integer, epoch integer, mode string, name string, iou float, tp integer, fn integer, fp integer);", NULL, NULL, NULL);
}


void insert_eval_segmentation(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus)
{
  char *err=NULL;
  char sql[LENGTH];
  float iou = -1.0;
  if (eval.tp + eval.fp + eval.fn) {
    iou = eval.tp / (float)(eval.tp + eval.fp + eval.fn);
  }
  int epoch = iteration/(int)(images/(batch/gpus)-0.5);
  sprintf(sql, "INSERT INTO eval_seg (iteration, epoch, mode, name, iou, tp, fn, fp) values(%d, %d, '%s', '%s', %f, %d, %d, %d);",
	  iteration,
	  epoch,
	  mode,
	  name,
	  iou,
	  eval.tp,
	  eval.fn,
	  eval.fp);
  //printf("%s\n",sql);
  int ret = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }   	  
}



void create_eval_segmentation_per_image(sqlite3 *db)
{
  int ret = sqlite3_exec(db,  "CREATE TABLE eval_seg_per_image (iteration integer, epoch integer, image_name, string, mode string, name string, iou float, tp integer, fn integer, fp integer);", NULL, NULL, NULL);
}


void insert_eval_segmentation_per_image(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus, char *image_name)
{
  char *err=NULL;
  char sql[LENGTH];  
  float iou;
  if ((eval.tp + eval.fp + eval.fn) == 0) {
    iou = -1.0;
  } else {
    iou = eval.tp / (float)(eval.tp + eval.fp + eval.fn);
  }
  int epoch = iteration/(int)(images/(batch/gpus)-0.5);
  sprintf(sql, "INSERT INTO eval_seg_per_image (iteration, epoch, image_name, mode, name, iou, tp, fn, fp) values(%d, %d, '%s', '%s', '%s', %f, %d, %d, %d);",
	  iteration,
	  epoch,
	  image_name,
	  mode,
	  name,
	  iou,
	  eval.tp,
	  eval.fn,
	  eval.fp);
  //  printf("%s\n",sql);
  int ret = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }   	  

}



void create_eval_detection_per_image(sqlite3 *db)
{
  int ret = sqlite3_exec(db,  "CREATE TABLE eval_det_per_image (iteration integer, epoch integer, image_name, string, mode string, name string, iou float, tp integer, fn integer, fp integer);", NULL, NULL, NULL);
}


void insert_eval_detection_per_image(sqlite3 *db, Eval_segmentation eval, char *mode, char *name, int iteration, int images, int batch, int gpus, char *image_name)
{
  char *err=NULL;
  char sql[LENGTH];  
  float iou;
  if ((eval.tp + eval.fp + eval.fn) == 0) {
    iou = -1.0;
  } else {
    iou = eval.tp / (float)(eval.tp + eval.fp + eval.fn);
  }
  int epoch = iteration/(int)(images/(batch/gpus)-0.5);
  sprintf(sql, "INSERT INTO eval_det_per_image (iteration, epoch, image_name, mode, name, iou, tp, fn, fp) values(%d, %d, '%s', '%s', '%s', %f, %d, %d, %d);",
	  iteration,
	  epoch,
	  image_name,
	  mode,
	  name,
	  iou,
	  eval.tp,
	  eval.fn,
	  eval.fp);
  //  printf("%s\n",sql);
  int ret = sqlite3_exec(db, sql, NULL, NULL, &err);
  if (SQLITE_OK != ret){
    printf("%s\n", err);
  }   	  

}

