from .fedavg import *
from .local import *
from .hyperfl import *
from .dp_fedavg import *
from .ppsgd import *
from .centaur import *


def local_update(rule):
    LocalUpdate = {'FedAvg': LocalUpdate_FedAvg,
                   'Local': LocalUpdate_StandAlone,
                   'HyperFL': LocalUpdate_HyperFL,
                   'DPFedAvg': LocalUpdate_DPFedAvg,
                   'PPSGD': LocalUpdate_PPSGD,
                   'CENTAUR': LocalUpdate_DPFedRep,
                   }

    return LocalUpdate[rule]
