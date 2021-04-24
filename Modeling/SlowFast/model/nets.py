import tensorflow as tf
from . import slowfast
from tensorflow.keras.layers import Input

__all__=['network']

def resnet50(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 6, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet101(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 4, 23, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet152(inputs, **kwargs):
    model = slowfast.SlowFast_body(inputs, [3, 8, 36, 3], slowfast.bottleneck, **kwargs)
    return model

def resnet200(inputs, **kwargs):
    model = slowfast.Slow_body(inputs, [3, 24, 36, 3], slowfast.bottleneck, **kwargs)
    return model



network = {
    'resnet50':resnet50,
    'resnet101':resnet101,
    'resnet152':resnet152,
    'resnet200':resnet200
}




