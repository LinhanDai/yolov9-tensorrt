import os
import tensorrt as trt


class TensorrtBase(object):
    def __init__(self, logger):
        '''
        Initialize the base class for building tensorrt
        :param logger: Logging system
        '''
        self.logger = logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.quantization_infer = None
        self.engine_file = None
        self.onnx_file = None

    def create_engine_if_not_exit(self):
        '''
        If the inference engine does not exist, create it
        :return: Whether the inference engine was successfully created
        '''
        serialized_model = None
        if os.path.exists(self.engine_file):
            return True
        else:
            builder = trt.Builder(self.trt_logger)
            config = builder.create_builder_config()
            engine = self.create_engine(builder, config)
            assert serialized_model is None, self.logger.error("engine create failure!")
            with open(self.engine_file, "wb") as f:
                f.write(engine.serialize())
        return True

    def create_engine(self, builder, config):
        '''
        创建推理引擎
        :param builder: TRT construction
        :param config:  TRT configuration
        :return:        Inference engine
        '''
        explicitBatch = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        network = builder.create_network(explicitBatch)
        parser = trt.OnnxParser(network, self.trt_logger)
        parsed = parser.parse_from_file(self.onnx_file)
        config.max_workspace_size = 1 << 30
        if self.quantization_infer == "FP16":
            self.logger.info("create engine with FP16")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            self.logger.info("create engine with TF32")
            config.set_flag(trt.BuilderFlag.TF32)

        input_Dims = network.get_input(0).shape
        if input_Dims[0] == -1:
            profile_calib = builder.create_optimization_profile()
            input_name = network.get_input(0).get_name()
            batch_dim = input_Dims
            batch_dim.d[0] = 1
            profile_calib.set_shape(input_name, batch_dim)
            config.add_optimization_profile(profile_calib)

        self.logger.info("Creating an inference engine, please wait a few minutes!!!")
        engine = builder.build_engine(network, config)
        self.logger.info("Creating an inference engine successful!")
        return engine
