# Training

## Prepare the data

### directory structure

Your need to prepare training data and validation data in darknet format. The directory structure should be like this:

```
base_dir
├── train
│   ├── JPEGImages
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   ├── 000003.jpg
|   |   ...
│   ├── labels
│   │   ├── 000001.txt
│   │   ├── 000002.txt
│   │   ├── 000003.txt
|   |   ...
└── val
    ├── JPEGImages
    │   ├── 000001.jpg
    │   ├── 000002.jpg
    │   ├── 000003.jpg
    |   ...
    ├── labels
    │   ├── 000001.txt
    │   ├── 000002.txt
    │   ├── 000003.txt
    |   ...
```

All the images should be in `JPEGImages` directory, and all the labels should be in `labels` directory. The label file name should be the same as the image file name, except the extension name. For example, the label file for `000001.jpg` should be `000001.txt`.

### label format

The label format is the same as darknet. Each line in the label file represents a bounding box for the corresponding image. The format is:

```shell
<object-class> <x> <y> <width> <height>
```

- object-class: the class index of the object in the image, starts from 0
- x, y: the center of the bounding box, normalized by the width and height of the image
- width, height: the width and height of the bounding box, normalized by the width and height of the image
- Note: the coordinates of the bounding box are normalized by the width and height of the image, so they should be in the range [0, 1]

### convert to darknet format

In many cases, the annotation of open dataset is not in darknet format. You can write a script to convert the dataset to darknet format.  
*If there are bbox outside the image boundary, you need to handle it.*

```python
def convert_bbox(img_wh, bbox):
    (width, height) = img_wh
    (x, y, w, h) = bbox
    # handle bbox outside the image boundary
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > width:
        w = width - x
    if y + h > height:
        h = height - y

    x /= width
    y /= height
    w /= width
    h /= height
    x += w / 2
    y += h / 2
    return x, y, w, h

def validate_bbox(bbox):
    (x, y, w, h) = bbox
    if w <= 0 or h <= 0:
        return False
    if x <= 0 or y <= 0:
        return False
    return True
```

### generate train.lst and val.lst

After you prepare the data, you need to generate `train.lst` and `val.lst` for training and validation. Each line in the `train.lst` and `val.lst` represents an image. The format is:

```shell
<image-path> # full path is recommended
<image-path>
<image-path>
...
```

You can use the following script to generate `train.lst` and `val.lst`:

```shell
base_dir=/path/to/your/data
find $base_dir/train/JPEGImages -name "*.jpg" > $base_dir/train/train.lst
find $base_dir/val/JPEGImages -name "*.jpg" > $base_dir/val/val.lst
```

### ~.names and ~.data

~.names is the class name file, each line represents a class name. List all the class names from class 0. 

~.data is the dataset config file, the format is:

```shell
classes = class_num
train = /path/to/train.lst
valid = /path/to/val.lst
names = /path/to/~.names
backup = /path/to/training/backup
```

Find example in `data/bdd100k.names` and `data/bdd100k.data`.


## Config the training

Write a config file for training. Find example in `cfg` dir.

Read [darknet wiki](https://github.com/AlexeyAB/darknet/wiki) for more details.

### cfg for custom dataset

<!-- TODO -->
Under construction...


## Start training

Train from scratch

```shell
./darknet detector train data/bdd100k.data cfg/yolov3-bdd100k.cfg -map -dont_show -clear
```

Resume training

```shell
./lightNet detector train data/bdd100k.data cfg/lightNet-BDD100K-1280x960.cfg lightNet-BDD100K-1280x960.weights -map -dont_show
```