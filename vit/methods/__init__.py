from .fedavg import *
from .local import *
from .hyperfl_lpm import *


def local_update(rule):
    LocalUpdate = {'FedAvg': LocalUpdate_FedAvg,
                   'FedAvg-Adapter': LocalUpdate_FedAvg,
                   'Local': LocalUpdate_StandAlone,
                   'Local-Adapter': LocalUpdate_StandAlone,
                   'HyperFL-LPM': LocalUpdate_HyperFL_LPM,
                   }

    return LocalUpdate[rule]
