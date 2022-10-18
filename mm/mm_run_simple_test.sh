import datetime
import numpy as np
import os
import os.path as osp
import cv2
import sys
import glob

np.set_printoptions(suppress=True, 
  precision=10,
  threshold=2000,
  linewidth=150)

import magicmind.python.runtime as mm
from magicmind.python.runtime import ModelKind, Network
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType

import sys
sys.path.append('../')
from utils.datasets import LoadImages

img_file = "../data/images/"
model_file="mm_magicmind_models/grid_tong_best.force_float16.magicmind"
mm_model=mm.Model()
mm_model.deserialize_from_file(model_file)
device_id = 0

# 打开设备
with mm.System() as mm_sys:
    dev_count = mm_sys.device_count()
    print("Device count: ", dev_count)
    if device_id >= dev_count:
        print("Invalid DEV_ID set!")
        assert device_id < dev_count
    # 打开MLU设备
    dev = mm.Device()
    dev.id = device_id
    assert dev.active().ok()
    print("Wroking on MLU ", device_id)

    # crete engine, context and queue
    engine = mm_model.create_i_engine()
    assert engine is not None
    context = engine.create_i_context()
    assert context is not None
    queue = dev.create_queue()
    assert queue is not None

    context = context
    queue = queue

    # yolov7，前处理图片处理，imread hwc 经过处理变成 nchw [1, 3, 640, 640]
    # torch 模型的输入是 nchw --》 # mm的模型也要是nchw (mm由onnx转，onnx的输入是 nchw)，不需要另外加

    dataset = LoadImages(img_file, img_size=640, stride=32)
    for path, img, im0s, vid_cap in dataset:
      print(path)
      
      import torch
      img = torch.from_numpy(img).to('cpu')
      img = img.float() 
      img /= 255.0  # 0 - 255 to 0.0 - 1.0
      if img.ndimension() == 3:
          img = img.unsqueeze(0)

      # 创建输⼊ inputs
      inputs = context.create_inputs()
      inputs[0].from_numpy(np.float32(img))

      # 创建输出 tensors
      outputs = []
      
      assert context.enqueue(inputs, outputs, queue).ok()
      assert queue.sync().ok()
      
      net_outs = np.array(outputs[0].asnumpy())

      print(net_outs)
