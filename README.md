
# Train-automation

This an automation for computer vision classification training that help you to: 
  - preprocesse image with (rescale/resize).
  - train model for wathever model architecture from tensorflow.keras.applications.  
  
  
```
python train.py --data data/ --model_name /model/model.h5 --size 128 --rescale --batch_size 64 --epochs 10
```
```
  --data: Directory of images to train
  --model_name: path to save model
  --size: size of Images
  --rescale: rescale the image
```
   
# # 

# Test-automation

This an automation for computer vision classification testing that help you to: 
  - preprocesse image with (greyscale/rescale/resize) 
  - predict one image classe.
  - evaluate the test set for the chosen model by multiple metrics(categorical crossentropy/F1 score/accuracy/recall/precision).   


## Image Directory Testing:
```
python auto_test_Dir.py --testdir Dir_test --model model.h5 --resize 128 --rescale
```
```
  --testdir: Directory of images to test
  --model: model path
  --resize: new size
  --rescale: rescale the image
  --greyscale: greyscale the image
```

## The MIT License

```
MIT
License: MIT
```
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

