from .detect import Y5rkDetector
from .vertex import BVTVertex
from .recognition import PPRCNNRecognition
from .common.tools_process import align_box
from .pipeline import LPRPipeline

_det_maps_ = dict(
    Y5rkDetector=Y5rkDetector,
)

_vertex_maps_ = dict(
    BVTVertex=BVTVertex,
)

_rec_maps_ = dict(
    PPRCNNRecognition=PPRCNNRecognition,
)


def build_pipeline(det_option: dict, vertex_option: dict, rec_option):
    det_option_ = det_option.copy()
    det_name = det_option_.pop("name")
    detector = _det_maps_[det_name](**det_option_)
    vertex_option_ = vertex_option.copy()
    vertex_name = vertex_option_.pop("name")
    vertex_predictor = _vertex_maps_[vertex_name](**vertex_option_)
    rec_option_ = rec_option.copy()
    rec_name = rec_option_.pop("name")
    recognizer = _rec_maps_[rec_name](**rec_option_)

    return LPRPipeline(detector=detector, vertex_predictor=vertex_predictor, recognizer=recognizer)
