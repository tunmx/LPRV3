from .pipeline import LPRPipeline


def det_maps(name: str):
    if name == "Y5rkDetectorORT":
        from .detect import Y5rkDetectorORT
        return Y5rkDetectorORT
    elif name == "Y5rkDetectorMNN":
        from .detect import Y5rkDetectorMNN
        return Y5rkDetectorMNN
    else:
        raise NotImplemented


def vertex_maps(name: str):
    if name == "BVTVertexORT":
        from .vertex import BVTVertexORT
        return BVTVertexORT
    if name == "BVTVertexMNN":
        from .vertex import BVTVertexMNN
        return BVTVertexMNN
    else:
        raise NotImplemented


def rec_maps(name: str):
    if name == "PPRCNNRecognitionORT":
        from .recognition import PPRCNNRecognitionORT
        return PPRCNNRecognitionORT
    elif name == "PPRCNNRecognitionMNN":
        from .recognition import PPRCNNRecognitionMNN
        return PPRCNNRecognitionMNN
    else:
        raise NotImplemented


def build_pipeline(det_option: dict, vertex_option: dict, rec_option):
    det_option_ = det_option.copy()
    det_name = det_option_.pop("name")
    detector = det_maps(det_name)(**det_option_)
    vertex_option_ = vertex_option.copy()
    vertex_name = vertex_option_.pop("name")
    vertex_predictor = vertex_maps(vertex_name)(**vertex_option_)
    rec_option_ = rec_option.copy()
    rec_name = rec_option_.pop("name")
    recognizer = rec_maps(rec_name)(**rec_option_)

    return LPRPipeline(detector=detector, vertex_predictor=vertex_predictor, recognizer=recognizer)
