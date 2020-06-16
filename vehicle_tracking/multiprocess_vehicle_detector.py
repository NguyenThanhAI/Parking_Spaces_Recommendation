import sys
import os
from collections import OrderedDict
from ctypes import c_bool
from multiprocessing import Process, Queue, Value
import numpy as np
import cv2
from mrcnn import config
from mrcnn.model import MaskRCNN
from vehicle_tracking.vehicle_detection import VehicleDetection

import torch
import torch.backends.cudnn as cudnn
from yolact.yolact import Yolact
from yolact.utils.functions import SavePath
from yolact.data import cfg, set_cfg
from yolact.layers.output_utils import postprocess
from yolact.utils.augmentations import FastBaseTransform

from code_timing_profiling.profiling import profile
from code_timing_profiling.timing import timethis


class MaskRCNNConfig(config.Config):
    NAME = "coco_pretrained_model_config"
    IMAGES_PER_GPU = 1
    GPU_COUNT = 1
    NUM_CLASSES = 1 + 1
    DETECTION_MIN_CONFIDENCE = 0.0


ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)


class MultiProcessVehicleDetector(object):
    def __init__(self, checkpoint_name="mask_rcnn_cars_and_vehicles_0008.h5", detection_vehicle_thresh=0.4, model_arch="mask_rcnn"):
        self.model_arch = model_arch
        self.checkpoint_name = checkpoint_name
        self.detection_vehicle_thresh = detection_vehicle_thresh

        self.in_queue = Queue(maxsize=1)
        self.out_queue = Queue(maxsize=1)
        self.stopped = Value(c_bool, False)

    @staticmethod
    def _run_function(in_queue, out_queue, stopped, model_arch, checkpoint_name, detection_vehicle_thresh):

        if model_arch.lower() == "mask_rcnn":

            PRETRAINED_DIR = os.path.join(ROOT_DIR, "test_object_detection_models")

            PRETRAINED_PATH = os.path.join(PRETRAINED_DIR, checkpoint_name)

            LOG_DIR = os.path.join(PRETRAINED_DIR, "logs")

            model = MaskRCNN(mode="inference", config=MaskRCNNConfig(), model_dir=LOG_DIR)

            model.load_weights(filepath=PRETRAINED_PATH, by_name=True)

        elif model_arch.lower() == "yolact":

            PRETRAINED_PATH = os.path.join(ROOT_DIR, "yolact", "weights", checkpoint_name)

            model_path = SavePath.from_str(PRETRAINED_PATH)

            config = model_path.model_name + "_config"

            print("Config not specified. Parsed %s from the file name.\n" % config)

            set_cfg(config)

            if torch.cuda.is_available():
                cudnn.fastest = True
                torch.set_default_tensor_type("torch.cuda.FloatTensor")
            else:
                torch.set_default_tensor_type("torch.FloatTensor")

            model = Yolact()
            model.load_weights(PRETRAINED_PATH)
            if torch.cuda.is_available():
                model.cuda()
            model.eval()
            model.detect.use_fast_nms = True
            model.detect.use_cross_class_nms = True
            cfg.mask_proto_debug = False
            cfg.eval_mask_branch = True

        if model_arch.lower() == "mask_rcnn":
            while not stopped.value:
                fid, frame = in_queue.get()
                if fid is None:
                    break

                rgb_frame = frame[:, :, ::-1]

                results = model.detect([rgb_frame], verbose=0)

                result = results[0]

                rois, scores, class_ids, masks = result["rois"], result["scores"], result["class_ids"], result["masks"]

                masks = np.transpose(masks, axes=(2, 0, 1))

                # make empty queue
                try:
                    out_queue.get(False)
                except:
                    pass

                out_queue.put((rois, scores, class_ids, masks))
        elif model_arch.lower() == "yolact":
            with torch.no_grad():
                while not stopped.value:
                    fid, frame = in_queue.get()
                    if fid is None:
                        break

                    if torch.cuda.is_available():
                        tensor_frame = torch.from_numpy(frame).cuda().float()
                    else:
                        tensor_frame = torch.from_numpy(frame).float()

                    batch = FastBaseTransform()(tensor_frame.unsqueeze(0))
                    preds = model(batch)
                    h, w, _ = tensor_frame.shape
                    save = cfg.rescore_bbox
                    cfg.rescore_bbox = True

                    t = postprocess(preds, w, h, visualize_lincomb=False, crop_masks=True, score_threshold=detection_vehicle_thresh)

                    cfg.rescore_bbox = save

                    class_ids, scores, rois, masks = [x.cpu().numpy() for x in t]

                    try:
                        out_queue.get(False)
                    except:
                        pass

                    out_queue.put((rois, scores, class_ids, masks))

    def start(self):
        self.process = Process(target=MultiProcessVehicleDetector._run_function,
                               args=(self.in_queue, self.out_queue, self.stopped, self.model_arch, self.checkpoint_name, self.detection_vehicle_thresh))
        self.process.daemon = True
        self.process.start()

    def stop(self):
        self.stopped.value = True
        try:
            self.in_queue.put((None, None, None, None), False)
        except Exception:
            pass
        self.process.join()

    def put_frame(self, frame_id, img):
        try:
            self.in_queue.get(False)
        except Exception:
            pass
        self.in_queue.put((frame_id, img))

    def get_result(self, block=True):
        if block:
            return self.out_queue.get()
        else:
            try:
                res = self.out_queue.get(False)
            except Exception:
                return None
            return res

    def warm_up(self):
        img = np.zeros((720, 1280, 3), np.uint8)
        self.put_frame(-1, img)
