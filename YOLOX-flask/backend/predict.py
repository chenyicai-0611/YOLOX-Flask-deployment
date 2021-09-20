#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

#import argparse
import os
import time
from loguru import logger

import cv2
import torch
import json
from pathlib import Path

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis
from backend.flask_id2name import id2name

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class DetectedInfo:
    boxes_detected = []

def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        if self.device == "gpu":
            img = img.cuda()

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre, self.nmsthre
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        DetectedInfo.boxes_detected = []  # 检测结果
        for i in range(len(bboxes)):
            box = bboxes[i]
            cls_id = int(cls[i])
            score = scores[i]
            if score < cls_conf:
                continue
            x0 = max(int(box[0]), 0)
            y0 = max(int(box[1]), 0)
            x1 = max(int(box[2]), 0)
            y1 = max(int(box[3]), 0)
            DetectedInfo.boxes_detected.append({"name": id2name[cls_id],
                               "conf": str(score.item()),
                               "bbox": [x0, y0, x1, y1]
                               })
        print('boxes_detected = ', DetectedInfo.boxes_detected)
        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if os.path.isdir(path):
        files = get_image_list(path)
    else:
        files = [path]
    files.sort()
    for image_name in files:
        outputs, img_info = predictor.inference(image_name)
        result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if save_result:
            save_folder = os.path.join(
                vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
            )
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break


def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args['path'] if args['demo'] == "video" else args['camid'])
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    save_folder = os.path.join(
        vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    )
    os.makedirs(save_folder, exist_ok=True)
    if args.demo == "video":
        save_path = os.path.join(save_folder, args.path.split("/")[-1])
    else:
        save_path = os.path.join(save_folder, "camera.mp4")
    logger.info(f"video save_path is {save_path}")
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )
    while True:
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            break

def preprocess_and_load():
    # 读取flask配置
    with open('./backend/flask_config.json', 'r', encoding='utf8') as fp:
        args = json.load(fp)
        print('Flask Config : ', args)

    exp = get_exp(args['exp_file'], args['name'])

    if not args['experiment_name']:
        args['experiment_name'] = exp.exp_name

    file_name = os.path.join(exp.output_dir, args['experiment_name'])
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args['save_result']:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)
        args['vis_folder'] = vis_folder

    if args['trt']:
        args['device'] = "gpu"

    #logger.info("Args: {}".format(args))
    if args['conf'] is not None:
        exp.test_conf = args['conf']
    if args['nms'] is not None:
        exp.nmsthre = args['nms']
    if args['tsize'] is not None:
        exp.test_size = (args['tsize'], args['tsize'])

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args['device'] == "gpu":
        model.cuda()
    model.eval()

    ckpt_file = args['ckpt']
    logger.info("loading checkpoint")
    ckpt = torch.load(ckpt_file, map_location="cpu")
    # load the model state dict
    model.load_state_dict(ckpt["model"])
    logger.info("loaded checkpoint done.")

    if args['fuse']:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args['trt']:
        assert not args['fuse'], "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None
    return model, exp, trt_file, decoder, args

def predict_and_postprocess(model, exp, trt_file, decoder, args):
    print('start to predict the uploaded image ......')
    predictor = Predictor(model, exp, COCO_CLASSES, trt_file, decoder, args['device'], args['legacy'])
    current_time = time.localtime()
    if args['demo'] == "image":
        img_path = str(Path(args['source']) / Path("img4predict.jpg")) # 读取路径
        image_demo(predictor, args['vis_folder'], img_path, current_time, args['save_result'])
    elif args['demo'] == "video" or args['demo'] == "webcam":
        imageflow_demo(predictor, args['vis_folder'], current_time, args)

if __name__ == "__main__":
    model, exp, trt_file, decoder, args = preprocess_and_load()
    predict_and_postprocess(model, exp, trt_file, decoder, args)