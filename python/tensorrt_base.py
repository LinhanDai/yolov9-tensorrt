import os
import tensorrt as trt


class TensorrtBase(object):
    def __init__(self, logger):
        '''
        初始化构建tensorrt的基类
        :param logger: 日志系统
        '''
        self.logger = logger
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.quantization_infer = None
        self.engine_file = None
        self.onnx_file = None

    def create_engine_if_not_exit(self):
        '''
        如果推理引擎不存在就进行创建
        :return: 推理引擎是否创建成功
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
        :param builder: trt构建
        :param config:  trt配置
        :return:        推理引擎
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
