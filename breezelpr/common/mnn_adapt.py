import numpy as np
import MNN
from loguru import logger

class MNNAdapter(object):

    def __init__(self, model_path: str, input_shape: tuple, output_name=None,
                 dim_type: int = MNN.Tensor_DimensionType_Caffe):
        self.interpreter = MNN.Interpreter(model_path)
        self.session = self.interpreter.createSession()
        if output_name == None:
            self.input_tensor = self.interpreter.getSessionInput(self.session)
        elif type(output_name) == str:
            self.input_tensor = self.interpreter.getSessionInput(self.session, output_name)
        else:
            logger.error("error output name format")
        self.input_shape = input_shape
        self.dim_type = dim_type

    def inference(self, tensor: np.ndarray) -> np.ndarray:
        tensor = tensor.astype(np.float32)
        tmp_input = MNN.Tensor(self.input_shape, MNN.Halide_Type_Float, tensor, self.dim_type)
        self.input_tensor.copyFrom(tmp_input)
        self.interpreter.runSession(self.session)
        output_tensor = self.interpreter.getSessionOutput(self.session)
        res = np.array(output_tensor.getData())

        return res
