import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

# from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import numpy as np
import magicmind.python.runtime as mm
from magicmind.python.runtime import ModelKind, Network
from magicmind.python.runtime.parser import Parser
from magicmind.python.runtime import DataType

device_id = 0
model_file="mm/mm_magicmind_models/yolov7.qint8_mixed_float16.magicmind"

mm_model=mm.Model()
mm_model.deserialize_from_file(model_file)

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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


        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # # Load model
        # model = attempt_load(weights, map_location=device)  # load FP32 model
        # stride = int(model.stride.max())  # model stride
        ### mlu
        stride= 32

        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        # if trace:
        #     model = TracedModel(model, device, opt.img_size)

        # if half:
        #     model.half()  # to FP16

        # # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        # names = model.module.names if hasattr(model, 'module') else model.names
        # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
        
        ### mlu
        names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        colors = [[34, 154, 14], [150, 231, 14], [138, 93, 19], [195, 226, 110], [37, 54, 102], [27, 212, 132], [47, 16, 48], [55, 244, 76], [222, 103, 104], [16, 47, 106], [113, 234, 220], [35, 97, 223], [160, 0, 143], [79, 45, 247], [251, 232, 193], [53, 157, 6], [238, 212, 42], [67, 5, 156], [107, 164, 165], [114, 113, 215], [181, 113, 113], [201, 63, 176], [211, 83, 43], [119, 77, 233], [124, 98, 48], [34, 38, 227], [13, 51, 3], [46, 126, 99], [214, 31, 45], [114, 34, 30], [181, 174, 13], [161, 194, 170], [83, 252, 229], [45, 248, 208], [20, 32, 78], [188, 22, 24], [87, 209, 126], [19, 41, 47], [130, 89, 86], [243, 140, 207], [40, 156, 98], [226, 190, 119], [25, 27, 207], [39, 91, 125], [215, 15, 124], [209, 247, 145], [26, 145, 17], [92, 180, 100], [60, 114, 96], [1, 20, 65], [218, 42, 47], [148, 163, 166], [8, 6, 117], [165, 51, 141], [58, 245, 56], [174, 14, 49], [228, 93, 40], [223, 80, 227], [141, 240, 155], [90, 89, 1], [108, 135, 57], [127, 32, 226], [28, 238, 230], [228, 36, 7], [224, 214, 125], [159, 21, 104], [140, 178, 71], [183, 78, 161], [136, 46, 129], [83, 202, 171], [81, 44, 160], [207, 100, 32], [4, 243, 45], [55, 17, 47], [214, 85, 57], [108, 170, 46], [59, 215, 196], [171, 5, 30], [167, 112, 73], [3, 148, 239]]


        # # Run inference
        # if device.type != 'cpu':
        #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        # old_img_w = old_img_h = imgsz
        # old_img_b = 1

        

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # 创建输⼊ inputs
            inputs = context.create_inputs()
            # 创建输出 tensors
            outputs = []

            # # Warmup
            # if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            #     old_img_b = img.shape[0]
            #     old_img_h = img.shape[2]
            #     old_img_w = img.shape[3]
            #     for i in range(3):
            #         model(img, augment=opt.augment)[0]
            
            # warmup
            inputs[0].from_numpy(img.numpy())
            assert context.enqueue(inputs, outputs, queue).ok()
            assert queue.sync().ok()
            
            # 
            inputs[0].from_numpy(img.numpy())
            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                # pred = model(img, augment=opt.augment)[0]
                assert context.enqueue(inputs, outputs, queue).ok()
                assert queue.sync().ok()
                pred = torch.from_numpy(outputs[0].asnumpy())
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'image':
                        cv2.imwrite(save_path, im0)
                        print(f" The image with the result is saved in: {save_path}")
                    else:  # 'video' or 'stream'
                        if vid_path != save_path:  # new video
                            vid_path = save_path
                            if isinstance(vid_writer, cv2.VideoWriter):
                                vid_writer.release()  # release previous video writer
                            if vid_cap:  # video
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            else:  # stream
                                fps, w, h = 30, im0.shape[1], im0.shape[0]
                                save_path += '.mp4'
                            vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        vid_writer.write(im0)

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()
