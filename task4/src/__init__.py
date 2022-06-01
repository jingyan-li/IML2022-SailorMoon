from src.PretrainModel import PretrainEmbeddingNet
from src.PretrainModelSimple import PretrainEmbeddingNet as PretrainEmbeddingNetSimple

def get_pretrainmodel(run):
    pretrain_model = PretrainEmbeddingNet(**run['pretrain_model_cfg']) if run[
                                                                              'model_type'] == "" else PretrainEmbeddingNetSimple(
        **run['pretrain_model_cfg'])
    return pretrain_model