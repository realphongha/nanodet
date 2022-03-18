# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time

import cv2
from pandas import option_context
import torch

from nanodet.data.transform import Pipeline
from nanodet.model.arch import build_model
from nanodet.util import (
    cfg,
    load_config,
    load_model_weight,
)


class Predictor(object):
    def __init__(self, cfg, model_path, logger=None, device="cuda:0"):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg.model)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert

            model = repvgg_det_model_convert(model, deploy_model)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.pipeline, cfg.data.val.keep_ratio)

    def inference(self, img):
        img_info = {}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["id"] = 69 # whatever
        for k in img_info:
            img_info[k] = [img_info[k]]
        meta = dict(img_info=img_info, raw_img=img, img=img)
        meta = self.pipeline(None, meta, self.cfg.data.val.input_size)
        meta["img"] = (
            torch.from_numpy(meta["img"].transpose(2, 0, 1))
            .unsqueeze(0)
            .to(self.device)
        )
        meta["raw_img"] = [meta["raw_img"]]
        meta["warp_matrix"] = [meta["warp_matrix"]]
        with torch.no_grad():
            results = self.model.inference(meta, cpu=(self.device=="cpu"))
        return meta, results

    def visualize(self, dets, meta, class_names, score_thres, wait=0):
        time1 = time.time()
        self.model.head.show_result(
            meta["raw_img"], dets, class_names, score_thres=score_thres, show=True
        )
        print("viz time: {:.3f}s".format(time.time() - time1))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', 
                        type=str, 
                        default='weights/nanodet_plus_m_320.pth', 
                        help='path to weights file')
    parser.add_argument('--cfg', 
                        type=str, 
                        default='config/nanodet-plus-m_320_exam.yml', 
                        help='path to config file')
    parser.add_argument('--img', 
                        type=str, 
                        default='test.jpg', 
                        help='path to image file')
    parser.add_argument('--device', 
                        type=str, 
                        default='cuda:0', 
                        help='device')
    parser.add_argument('--test-speed', 
                        type=int, 
                        default=100, 
                        help='run n times to test speed')
    opt = parser.parse_args()
    
    load_config(cfg, opt.cfg)
    
    engine = Predictor(cfg, opt.weights, device=opt.device)
    
    img = cv2.imread(opt.img)
    
    meta, results = engine.inference(opt.img)
    results = results[69]
    for k in results:
        for res in results[k]:
            xmin, ymin, xmax, ymax, score = res
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img, "Class %i %.4f" % (k, score), (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("blah", cv2.resize(img, (640, 640)))
    cv2.waitKey()
    