FFHQ Dataset Preparation
========================

The source images were downloaded from the official FFHQ dataset page:

https://github.com/NVlabs/ffhq-dataset

64 x 64 dataset
----------------

We downloaded the FFHQ "thumbnails128x128" images and resized them to
64 x 64 using resize_ffhq.py:

    python resize_ffhq.py --img_size 64

The resized images are saved under the "ffhq64" directory.

256 x 256 dataset
-----------------

We downloaded the first 5,000 images from the FFHQ "images1024x1024"
dataset and resized them from 1024 x 1024 to 256 x 256 using
resize_ffhq.py:

    python resize_ffhq.py --img_size 256

The resized images are saved under the "ffhq256" directory. Only these
first 5,000 images were used for the 256 x 256 numerical experiments.

Place "thumbnails128x128" and "images1024x1024" in this directory before
running the commands. The script selects the appropriate source directory
from --img_size automatically. For --img_size 256, it processes the first
5,000 PNG files in filename order.

Please refer to LICENSE.txt and the official FFHQ page for the applicable
dataset license and terms of use.
