# Coco Dots

## About
COCO Dots is based on the original COCO panoptic segmentation dataset. It was designed to train models on a grouping task similar to the one used in a human psychophysics study by Jeurissen et al. (2016). The most important addition to the original COCO panoptic json-file is the serrelab_anns key

The format of the original annotation file can be found here: [COCO 2017 Annotations Format](https://cocodataset.org/#format-data)

We have modified the annotations file to include the following, highlighted fields:

![SerreLab Annotations](https://ibb.co/HrcFB8F)

## Usage
1. Download the COCO 2017 train and validation datasets:
  -  [2017 Train Images](http://images.cocodataset.org/zips/train2017.zip)
  -  [2017 Val Images](http://images.cocodataset.org/zips/val2017.zip)

2. Download the Annotation files:

3. Create the DataLoaders as follows:
  ```
  train_loader = torch.utils.data.DataLoader(
  CocoDots("<PATH_TO_TRAIN_ANNOTATIONS_JSON>", "<PATH_TO_COCO_2017_TRAIN_IMAGES>", conversion='WhiteOutline'),
  batch_size=128, num_workers=4, pin_memory=False, drop_last=True)

  val_loader = torch.utils.data.DataLoader(
  CocoDots("<PATH_TO_VAL_ANNOTATIONS_JSON>", "<PATH_TO_COCO_2017_VAL_IMAGES>", conversion='WhiteOutline'),
  batch_size=128, num_workers=4, pin_memory=False, drop_last=True)
  ```
  