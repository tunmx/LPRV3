import cv2
import numpy as np
from .base.base import HamburgerABC
from .common.tools_process import cost


class BVTVertex(HamburgerABC):

    def __init__(self, onnx_path, *args, **kwargs):
        import onnxruntime as ort
        super().__init__(*args, **kwargs)
        self.session = ort.InferenceSession(onnx_path, None)
        self.input_config = self.session.get_inputs()[0]
        self.output_config = self.session.get_outputs()[0]
        self.input_size = self.input_config.shape[2:]

    @staticmethod
    def encode_images(image: np.ndarray):
        image_encode = image / 255.0
        if len(image_encode.shape) == 4:
            image_encode = image_encode.transpose(0, 3, 1, 2)
        else:
            image_encode = image_encode.transpose(2, 0, 1)
        image_encode = image_encode.astype(np.float32)

        return image_encode

    @cost('Vertex')
    def _run_session(self, data) -> np.ndarray:
        result = self.session.run([self.output_config.name], {self.input_config.name: data})

        return result[0]

    def _postprocess(self, data) -> np.ndarray:
        assert data.shape[0] == 1
        data = np.asarray(data).reshape(-1, 4, 2)
        data[:, :, 0] *= self.input_size[1]
        data[:, :, 1] *= self.input_size[0]

        return data[0]

    def _preprocess(self, image) -> np.ndarray:
        assert len(
            image.shape) == 3, "The dimensions of the input image object do not match. The input supports a single " \
                               "image. "
        image_resize = cv2.resize(image, self.input_size)
        encode = self.encode_images(image_resize)
        encode = encode.astype(np.float32)
        input_tensor = np.expand_dims(encode, 0)

        return input_tensor