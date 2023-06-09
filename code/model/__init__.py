from model.abstract_model import AbstractRecModel
from model.mf import MF
from model.lightgcn import LightGCN
from model.sgl import SGL
from model.qkv import QKV
from model.graphcl import GraphCL
from model.kgcl import KGCL
from model.gat import GAT
from model.kgcl_my import KGCL_my
from model.ssm import SSM
from model.graphadd import GraphADD
from model.pcl import PCL
from model.simgcl import SimGCL
from model.kgin import KGIN


def get_model_by_name(name):
    model = eval(name)()
    return model
