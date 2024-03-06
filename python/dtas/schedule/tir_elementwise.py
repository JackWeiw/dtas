"""
for softmax, layernorm

"""
import numpy as np
from tvm import tir

from .tir_base import TIRSchedulerBase
from ..common.analisys import *
from ..common.utils import (
    save_to_file,
    get_root_block,
)
from ..common.config import ElementwiseConfig


def save_ir(filename, sch):
    with open(filename, "w") as f:
        f.write(sch.mod.script())


class TIRElementwiseScheduler(TIRSchedulerBase):
    def apply_config(self, config: ElementwiseConfig) -> Optional[tir.Schedule]:
        sch= self.func_info.create_schedule()
        root_block = get_root_block(sch)
        blocks = sch.get_child_blocks(root_block)
        assert len(blocks) == 1, "element wise op should have only one spatial block"
        l = sch.get_loops(blocks[0])
        bx, u, tx, vec = sch.split(
            l[0],
            [
                config.grid_size,
                None,
                config.len_tx,
                config.vector_size,
            ],
        )
        sch.bind(bx, "blockIdx.x")
        sch.annotate(
            u,
            ann_key="pragma_auto_unroll_max_step",
            ann_val=config.unroll_depth,
        )
        sch.annotate(u, ann_key="pragma_unroll_explicit", ann_val=1)
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        # print(sch.mod)
        # save_to_file(
        #     f"/home/weitao/XIAG8XX/profile/testIR/Elementwise/ir/{config.len_bx}.py",
        #     sch,
        # )
        return sch
