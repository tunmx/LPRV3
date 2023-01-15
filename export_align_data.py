import cv2
import tqdm

import breezelpr as bpr
import click
import numpy as np
import yaml
from easydict import EasyDict as edict
from utils.visual_tools import *
import os


def load_cfg(config_path: str) -> edict:
    with open(config_path) as f:
        data_dict = yaml.load(f, Loader=yaml.FullLoader)
    cfg = edict(data_dict)

    return cfg


@click.command()
@click.option("-config", "--config", default="config/planB_mnn.yml", type=click.Path(exists=True))
@click.option("-src", "--src", type=click.Path(exists=True))
@click.option("-dst", "--dst", type=click.Path())
def run(config, src, dst):
    cfg = load_cfg(config)
    print(cfg)
    build_pipeline_option = cfg.build_pipeline
    lpr_predictor = bpr.build_pipeline(**build_pipeline_option)
    os.makedirs(dst, exist_ok=True)
    images = [os.path.join(src, item) for item in os.listdir(src) if
                     item.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']]
    for idx, path in enumerate(tqdm.tqdm(images)):
        image = cv2.imread(path)
        result = lpr_predictor(image)
        print(result)
        # canvas = draw_full(image, result)
        for res in result[:1]:
            vertex = res['vertex']
            crop = bpr.get_rotate_crop_image(image, vertex)
            print(os.path.join(dst, os.path.basename(src)))
            cv2.imwrite(os.path.join(dst, os.path.basename(path)), crop)
            # cv2.imshow("w", crop)
            # cv2.waitKey(0)


if __name__ == '__main__':
    run()
