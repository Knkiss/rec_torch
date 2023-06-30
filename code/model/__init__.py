from model.abstract_model import AbstractRecModel
from model.general.mf import MF
from model.general.lightgcn import LightGCN
from model.general.sgl import SGL
from model.mine.qkv import QKV
from model.mine.graphcl import GraphCL
from model.knowledge.kgcl import KGCL
from model.knowledge.gat import GAT
from model.mine.kgcl_my import KGCL_my
from model.general.ssm import SSM
from model.mine.graphadd import GraphADD
from model.mine.pcl import PCL
from model.general.simgcl import SimGCL
from model.knowledge.kgin import KGIN
from model.knowledge.kgat import KGAT
from model.knowledge.kgcn import KGCN


def get_model_by_name(name):
    model = eval(name)()
    return model
