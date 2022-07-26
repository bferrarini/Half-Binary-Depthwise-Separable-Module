# How to execute a LCE model on the RPI4

We provide the binary *lce_cnn*. This tool can run and benchamrck any model comepiled for LCE ot TFLITE on an ARM64 platform.
The source code is provided in _src_.

```
./lce_cnn -h
```

### Computing an image representation

```
./lce_cnn -g HB12-FN.tflite -i image000.jpg -o feature_file.txt
```

### Benchmarking a model

```
./lce_cnn -g HB12-FN.tflite -b
```

## FloppyNet

For FloppyNet, you can download the model we used for our paper from https://github.com/bferrarini/FloppyNet_TRO/tree/master/FloppyNet_TRO/TRO_pretrained.
