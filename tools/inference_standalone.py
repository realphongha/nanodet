import os
import cv2
import numpy as np
from .standalone_engine import NanoDetMnn, NanoDetOnnx


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='weights/nanodet_plus_m_320.onnx', 
                        help='path to weights file')
    parser.add_argument('--engine', 
                        type=str, 
                        default='onnx', 
                        help='inference engine')
    parser.add_argument('--img', 
                        type=str, 
                        default='test.jpg', 
                        help='path to image file')
    parser.add_argument('--imgsz', 
                        nargs='+', 
                        type=int, 
                        default=[320], help='input size h,w')
    parser.add_argument('--mean', 
                        nargs='+', 
                        type=float, 
                        default=[103.53, 116.28, 123.675], help='img mean for normalization')
    parser.add_argument('--std', 
                        nargs='+', 
                        type=float, 
                        default=[57.375, 57.12, 58.395], help='img std for normalization')
    parser.add_argument('--num-cls',
                        type=int, 
                        default=5, help='number of classes')
    parser.add_argument('--iou-thres',
                        type=float, 
                        default=0.2, help='iou threshold')
    parser.add_argument('--score-thres', 
                        nargs='+', 
                        type=float, 
                        default=[0.5, 0.2, 0.2, 0.2, 0.2], 
                        help='classwise score threshold')
    parser.add_argument('--reg-max', 
                        type=int, 
                        default=7, help='reg max')
    parser.add_argument('--strides', 
                        nargs='+', 
                        type=int, 
                        default=[8, 16, 32, 64], help='for postprocessing')
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda:0', 
                        help='device')
    parser.add_argument('--test-speed', 
                        type=int, 
                        default=None, 
                        help='run n times to test speed')
    opt = parser.parse_args()
    if len(opt.imgsz) == 1:
        opt.imgsz *= 2
    opt.imgsz = tuple(opt.imgsz)
    assert opt.num_cls == len(opt.score_thres)
    
    if opt.engine.lower() == "onnx":
        engine = NanoDetOnnx(opt.weights, opt.device, opt.imgsz, 
                             opt.std, opt.mean,
                             opt.num_cls, opt.reg_max, opt.strides,
                             opt.score_thres, opt.iou_thres)
    elif opt.engine.lower() == "mnn":
        engine = NanoDetMnn(opt.weights, opt.device, opt.imgsz, 
                            opt.std, opt.mean,
                            opt.num_cls, opt.reg_max, opt.strides,
                            opt.score_thres, opt.iou_thres)
    else:
        raise NotImplementedError("%s is not implemented!" % opt.engine)
    img = cv2.imread(opt.img)
    h, w = img.shape[:2]
    if opt.test_speed:
        for _ in range(opt.test_speed):
            bboxes = engine.infer(img.copy())
    else:
        bboxes = engine.infer(img.copy())
    for bbox in bboxes:
        # print(bbox)
        xmin, ymin, xmax, ymax, score, cls = bbox
        xmin = int(xmin / opt.imgsz[1] * w)
        xmax = int(xmax / opt.imgsz[1] * w)
        ymin = int(ymin / opt.imgsz[0] * h)
        ymax = int(ymax / opt.imgsz[0] * h)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(img, "Class %i %.4f" % (cls, score), (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 2, cv2.LINE_AA)
    img = cv2.resize(img, (640, 640))
    cv2.imshow("blah", img)
    cv2.waitKey()
    