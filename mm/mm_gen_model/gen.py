from operator import eq
import os
import json
import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType
import argparse
from calibrator import CalibData
import sys
sys.path.append('../../')
from utils.datasets import LoadImages

def onnx_parse(args):
    # 创建MagicMind parser
    parser = Parser(mm.ModelKind.kOnnx)
    # 使用parser将Caffe模型文件转换为MagicMind Network实例。
    network = mm.Network()
    assert parser.parse(network, args.onnx_model).ok()

    # 设置网络输入数据形状
    input_dims = mm.Dims((args.batch_size, 3, args.input_height, args.input_width))
    assert network.get_input(0).set_dimension(input_dims).ok()
    return network

def generate_model_config(args):
    config = mm.BuilderConfig()

    # 指定硬件平台
    assert config.parse_from_string('{"archs":[{"mtp_372": [6,8]}]}').ok()
    assert config.parse_from_string('{"opt_config":{"type64to32_conversion":true}}').ok()
    assert config.parse_from_string('{"opt_config":{"conv_scale_fold":true}}').ok()
    assert config.parse_from_string("""{"cross_compile_toolchain_path": "/tmp/gcc-linaro-6.2.1-2016.11-x86_64_aarch64-linux-gnu/"}""").ok()
    # assert config.parse_from_string('{"computation_preference":fast}').ok()
    # config.parse_from_string('{"computation_preference":"fast"}')
    
    # 输入数据摆放顺序
    # 输入顺序的改变需要同步到推理过程中的网络预处理实现，保证预处理结果的输入顺序与网络输入数据顺序一致。
    # 以下JSON字符串中的0代表改变的是网络第一个输入的数据摆放顺序。1则代表第二个输入，以此类推。
    # assert config.parse_from_string('{"convert_input_layout": {"0": {"src": "NCHW", "dst": "NHWC"}}}').ok()
    
    # 模型输入输出规模可变功能
    if args.shape_mutable == "true":
        assert config.parse_from_string('{"graph_shape_mutable":true}').ok()
        assert config.parse_from_string('{"dim_range": {"0": {"min": [1, 3, 288, 288], "max": [1, 3, 640, 640]}}}').ok()
    else:
        assert config.parse_from_string('{"graph_shape_mutable":false}').ok()
    
    # 精度模式
    assert config.parse_from_string('{"precision_config":{"precision_mode":"%s"}}' % args.quant_mode).ok()

    # 量化算法，支持对称量化（symmetric)和非对称量化（asymmetric）。当量化统计算法设置为EQNM_ALOGORITHM时，仅适用于对称量化。
    assert config.parse_from_string('{"precision_config": {"activation_quant_algo": "symmetric"}}').ok()
    
    # # 设置量化粒度，支持按tensor量化（per_tensor）和按通道量化（per_axis）两种。
    # assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_tensor"}}').ok()
    assert config.parse_from_string('{"precision_config": {"weight_quant_granularity": "per_axis"}}').ok()
    
    return config

def calibrate(args, network : mm.Network, config : mm.BuilderConfig):
    # 创建量化工具并设置量化统计算法
    dataset = LoadImages(args.image_dir, img_size=640, stride=32)
    sample_data = []
    count = 0
    for path, img, im0s, vid_cap in dataset:
        import torch
        img = torch.from_numpy(img).to('cpu')
        img = img.float()
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        print("calibrate preprocess img.shape:", img.shape)
        sample_data.append(img)
        count = count + 1
        if count == 16:
            break
    print("break")
    calib_data = CalibData(sample_data)
    calibrator = mm.Calibrator([calib_data])
    assert calibrator is not None
    # 设置量化统计算法，支持线性统计算法（LINEAR_ALGORITHM）及加强的最小化量化噪声算法（EQM_ALGORITHM）。
    # assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.LINEAR_ALGORITHM).ok()
    assert calibrator.set_quantization_algorithm(mm.QuantizationAlgorithm.EQNM_ALGORITHM).ok()

    # 打开设备
    with mm.System() as mm_sys:
        dev_count = mm_sys.device_count()
        print("Device count: ", dev_count)
        if args.device_id >= dev_count:
            print("Invalid device set!")
            # 打开MLU设备
            dev = mm.Device()
            dev.id = args.device_id
            assert dev.active().ok()
            print("Wroking on MLU ", args.device_id)
    # 进行量化
    assert calibrator.calibrate(network, config).ok()

def main():
    args = argparse.ArgumentParser()
    # args.add_argument("--pytorch_model", "--pytorch_model", type=str, default="../pytorch_model/resnet50.trace.pth", help="jit trace resnet50 model")
    args.add_argument("--onnx_model", "--onnx_model", type=str, default="../onnx_model/yolov7.onnx", help="yolov7 onnx model")
    args.add_argument("--output_model", "--output_model", type=str, default="yolo.model", help="save mm model to this path")
    args.add_argument("--image_dir", "--image_dir",  type=str, default="/nfsdata/datasets/imageNet2012/", help="imagenet val datasets")
    args.add_argument("--quant_mode", "--quant_mode", type=str, default="qint8_mixed_float16", help="qint8_mixed_float16, force_float32, force_float16")
    args.add_argument("--shape_mutable", "--shape_mutable", type=str, default="false", help="whether the mm model is dynamic or static or not")
    args.add_argument('--batch_size', dest = 'batch_size', default = 1, type = int, help = 'batch_size')
    args.add_argument('--input_width', dest = 'input_width', default = 640, type = int, help = 'model input width')
    args.add_argument('--input_height', dest = 'input_height', default = 640, type = int, help = 'model input height')
    args.add_argument('--device_id', dest = 'device_id', default = 0, type = int, help = 'device_id')
    args = args.parse_args()

    supported_quant_mode = ['qint8_mixed_float16', 'qint8_mixed_float32', 'force_float16', 'force_float32']
    if args.quant_mode not in supported_quant_mode:
        print('quant_mode [' + args.quant_mode + ']', 'not supported')
        exit()
    
    # network = pytorch_parser(args)
    network = onnx_parse(args)
    
    config = generate_model_config(args)
    if args.quant_mode.find('qint') != -1:
        print('do calibrate...')
        calibrate(args, network, config)
    
    # 生成模型
    print('build model...')
    builder = mm.Builder()
    model = builder.build_model(args.output_model, network, config)

    assert model is not None
    # 将模型序列化为离线文件
    assert model.serialize_to_file(args.output_model).ok()
    print("Generate model done, model save to %s" % args.output_model)

if __name__ == "__main__":
    main()

