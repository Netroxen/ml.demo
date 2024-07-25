# Image Classifier

This is a minimal setup to train an image classifier using labelled images.


### Usage

This project includes some sample training data.

To adapt this project to other images, ensure that the following conditions are
met:

* A minimum of 50 images per category
* Place the images in their respective folders under `./v_data/train/categoryname`
* Place an equal amount of **different** images (of the same category) under `./v_data/test/categoryname`

```bash
pip install -r requirements.txt
```

Before running Tensorflow, ensure the images meet the necessary format
requirements by running:

```bash
python check.py
```

To begin training on your data, run the following:

```bash
python main.py
```

Before predicting an image, make sure that all labels are added to the `classes = []`
variable in `predict.py`.

To predict an image, change the path in `predict.py`.
