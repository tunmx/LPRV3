import cv2
import breezelpr as bpr
import click
import numpy as np
import yaml
from easydict import EasyDict as edict
from utils.visual_tools import *

det_option = dict(
    name="Y5rkDetector",
    onnx_path="resource/det/y5s_r_det_320x.onnx",
    box_threshold=0.5,
    nms_threshold=0.6
)
vertex_option = dict(
    name="BVTVertex",
    onnx_path="resource/vertex/vertex_mnet025_x96.onnx",
)
rec_option = dict(
    name="PPRCNNRecognition",
    onnx_path="resource/rec/rec_ptocr_v3_rec_infer.onnx",
    character_file="resource/rec/ppocr_keys_v1.txt"
)


def load_cfg(config_path: str) -> edict:
    with open(config_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(data_dict)

    return cfg


@click.command()
@click.option("-config", "--config", default="config/planA_y5srk_320_bvt_025_ppocr.yml", type=click.Path(exists=True))
@click.option("-image", "--image", type=click.Path(exists=True))
def run(config, image, ):
    cfg = load_cfg(config)
    print(cfg)
    build_pipeline_option = cfg.build_pipeline
    image = cv2.imread(image)
    lpr_predictor = bpr.build_pipeline(**build_pipeline_option)
    result = lpr_predictor(image)
    canvas = draw_full(image, result)

    cv2.imshow("canvas", canvas)
    cv2.waitKey(0)
    cv2.imwrite("out.jpg", canvas)


if __name__ == '__main__':
    run()
