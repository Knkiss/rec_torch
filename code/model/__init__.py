from model.abstract_model import AbstractRecModel
from model.general.mf import MF
from model.general.lightgcn import LightGCN
from model.general.sgl import SGL
from model.unuse.qkv import QKV
from model.unuse.graphcl import GraphCL
from model.knowledge.kgcl import KGCL
from model.knowledge.gat import GAT
from model.mine.kgic import KGIC
from model.general.ssm import SSM
from model.unuse.graphadd import GraphADD
from model.mine.pcl import PCL
from model.general.simgcl import SimGCL
from model.knowledge.kgin import KGIN
from model.knowledge.kgat import KGAT
from model.knowledge.kgcn import KGCN
from model.knowledge.cke import CKE
from model.knowledge.mcclk import MCCLK
from model.general.sgl_recbole import SGL_recbole
from model.GJJ.CGCL import CGCL
from model.knowledge.kgrec import KGRec
from model.general.graphda import GraphDA
from model.mine.work2 import WORK2
from model.general.xsimgcl import XSimGCL
from model.mine.work3 import WORK3
from model.mine.KGPro import KGPro


def get_model_by_name(name):
    model = eval(name)()
    return model
