"""Analysis on TIR blocks, loops and functions."""
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Set, Union, Dict, Tuple, Callable
from enum import Enum
from typing_extensions import Literal

import tvm
from tvm import ir, tir, arith
from tvm._ffi import get_global_func
from tvm.tir import PrimExpr, Var, IterVar
from tvm.ir import Range
from tvm.tir.analysis import undefined_vars

from .utils import (
    _is_one,
    detect_dominant_read,
    collect_block_iter_vars_used_in_access_region,
    inline_transpose_block,
    get_root_block,
    get_reduction_blocks,
    is_transpose_block,
    is_identity_block,
)
from ..logging import get_log_level, debug_info

TVM_DEFAULT_NAME = "default_function_kernel0"


class IterKind(Enum):
    """Iter kinds for GEMM-liked programs.
    We can simplify the computation to C[S, I, J] += A[S, I, K] * B[S, J, K],
    where `I, J, K` are fundamental axes for gemm and `S` represents all
    other spatial axes (e.g. batches)
    kIter_S: spatial axes
    kIter_I: I axes
    kIter_J: J axes
    kIter_K: K axes
    kIter_T: trivial axes (i.e. with extent 1)
    """

    kIter_S = 0
    kIter_I = 1
    kIter_J = 2
    kIter_K = 3
    kIter_T = 4


class FuncKind(Enum):
    kFunc_GEMM = 0
    kFunc_GEMV = 1
    kFunc_Reduction = 2
    kFunc_Elementwise = 3
    kFunc_Transpose = 4
    KFunc_Fallback = 5


@dataclass
class IterTrait:
    kind: IterKind
    extent: PrimExpr


class IterInfo:
    """Information about a loop/iter var."""

    kind: Literal["S", "R", "O"]
    var: tir.Var
    _dom: tir.PrimExpr
    loop_rv: tir.schedule.LoopRV

    def __init__(
        self,
        kind: Literal["S", "R", "O"],
        var: tir.Var,
        dom: tir.PrimExpr,
        loop_rv: tir.schedule.LoopRV,
    ):
        """Construct an IterInfo object."""
        self.kind = kind
        self.var = var
        self._dom = dom
        self.loop_rv = loop_rv

    @property
    def dom(self) -> Union[int, tir.PrimExpr]:
        """The iteration domain of the loop."""
        return int(self._dom) if isinstance(self._dom, tir.IntImm) else self._dom

    def __str__(self) -> str:
        return f'Iter("{self.kind}", {self.dom})'

    def __repr__(self) -> str:
        return str(self)


class BlockInfo:
    """Information about a TIR block."""

    sch: tir.Schedule
    name: str
    iters: List[IterInfo]
    block_rv: tir.schedule.BlockRV
    _reduction_block: bool

    def __init__(
        self,
        name: str,
        sch: tir.Schedule,
        iters: List[IterInfo],
        block_rv: tir.schedule.BlockRV,
        reduction_block: bool = False,
    ):
        """Construct a BlockInfo object."""
        self.name = name
        self.sch = sch
        self.block_rv = block_rv
        self.iters = iters
        self._reduction_block = reduction_block

    def dom(self) -> List[Union[int, tir.PrimExpr]]:
        """The iteration domain of the block."""
        return [i.dom for i in self.iters]

    def dom_kind(self) -> str:
        """The iteration domain kind of the block, for example, SSSS, SSSR."""
        return "".join(i.kind for i in self.iters)

    def is_injective(self) -> bool:
        """Whether the block is injective, i.e. all its iteration domains are injective."""
        return all(k == "S" for k in self.dom_kind())

    def is_elementwise(self, sch: tir.Schedule) -> bool:
        """Whether the block is elementwise, i.e. trivial mapping between read/write region"""
        # WT 这个判断有问题, 存在Full,没有read,只有write,
        def _check_unit_var_range(dom: ir.Range, var: tir.Var) -> bool:
            return dom.min.same_as(var) and dom.extent == 1

        if not self.is_injective():
            print("1")
            return False
        block = sch.get(self.block_rv)
        # if len(block.writes) != 1:
        #     print('2')
        #     return False
        r_region = [r.region for r in block.reads]
        w_region = block.writes[0].region
        """
    @T.prim_func(private=True)
    def divide1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(50280)), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(50280)):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[()])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[()]
    这个应该也是elementwise
        """
        # for r in r_region:
        #     if len(r) != len(w_region):
        #         print('3')
        #         return False
        # for var, r_doms, w_dom in zip(block.iter_vars, r_region, w_region):
        #     for r_dom in r_doms:
        #         if not _check_unit_var_range(r_dom, var) or not _check_unit_var_range(
        #             w_dom, var
        #         ):
        #             print('4')
        #             return False
        return True

    def is_transpose(self) -> bool:
        raise NotImplementedError

    def is_reduction(self) -> bool:
        """Whether the block is a reduction workload."""
        # TODO(@junrushao): distinguish GEMV and reduction
        return self._reduction_block

    def get_reduction_type(self) -> FuncKind:
        """
        distinguish GEMM, GEMV, Reduction
        """
        block_stmt = self.sch.get(self.block_rv)
        conditions = []
        conditions.append(self.is_reduction())
        conditions.append(len(block_stmt.reads) >= 2)
        conditions.append(len(block_stmt.writes) == 1)
        conditions.append(self._get_reduction_expr(block_stmt) is not None)
        conditions.append(
            len(
                collect_block_iter_vars_used_in_access_region(
                    block_stmt, block_stmt.writes[0].region
                )
            )
            > 0
        )
        if not all(conditions):
            return FuncKind.kFunc_Reduction

        iter_num = len(block_stmt.iter_vars)
        ret = [
            read.buffer
            for read in block_stmt.reads
            if len(
                collect_block_iter_vars_used_in_access_region(block_stmt, read.region)
            )
            < iter_num
            and len(
                collect_block_iter_vars_used_in_access_region(block_stmt, read.region)
            )
            > 0
        ]
        if 0 < len(ret) < len(block_stmt.reads):
            return FuncKind.kFunc_GEMV

        num_read_vars_A = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.reads[0].region
            )
        )
        num_read_vars_B = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.reads[1].region
            )
        )
        num_write_vars_C = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.writes[0].region
            )
        )
        if num_read_vars_A == num_read_vars_B == num_write_vars_C:
            return FuncKind.kFunc_GEMM
        return FuncKind.kFunc_Reduction

    def _get_reduction_expr(self, block: tir.Block) -> Optional[tir.PrimExpr]:
        # Detect and return `Y` in `X[...] = X[...] + Y`
        buffer_store = block.body
        if not isinstance(buffer_store, tir.BufferStore):
            return None
        if not isinstance(buffer_store.value, tir.Add):
            return None
        if not ir.structural_equal(
            buffer_store.value.a,
            tir.BufferLoad(buffer_store.buffer, block.body.indices),
            map_free_vars=True,
        ):
            return None
        return buffer_store.value.b

    def is_gemv(self) -> Optional[List[tir.Buffer]]:
        """Whether the block is a GEMV workload."""
        block_stmt = self.sch.get(self.block_rv)
        conditions = []
        conditions.append(self.is_reduction())
        conditions.append(len(block_stmt.reads) >= 2)
        conditions.append(len(block_stmt.writes) == 1)
        conditions.append(self._get_reduction_expr(block_stmt) is not None)
        conditions.append(
            len(
                collect_block_iter_vars_used_in_access_region(
                    block_stmt, block_stmt.writes[0].region
                )
            )
            > 0
        )
        if not all(conditions):
            return None
        iter_num = len(block_stmt.iter_vars)
        ret = [
            read.buffer
            for read in block_stmt.reads
            if len(
                collect_block_iter_vars_used_in_access_region(block_stmt, read.region)
            )
            < iter_num
            and len(
                collect_block_iter_vars_used_in_access_region(block_stmt, read.region)
            )
            > 0
        ]
        return ret if 0 < len(ret) < len(block_stmt.reads) else None

    def is_gemm(self) -> bool:
        """
        Whether the block is a GEMM workload.
        如果是broadcast gemm就没有考虑在里面
        """
        block_stmt = self.sch.get(self.block_rv)
        conditions = []
        conditions.append(self.is_reduction())
        conditions.append(len(block_stmt.reads) == 2)
        conditions.append(len(block_stmt.writes) == 1)
        conditions.append(self._get_reduction_expr(block_stmt) is not None)
        conditions.append(
            len(
                collect_block_iter_vars_used_in_access_region(
                    block_stmt, block_stmt.writes[0].region
                )
            )
            > 0
        )
        if not all(conditions):
            return False
        num_read_vars0 = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.reads[0].region
            )
        )
        num_read_vars1 = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.reads[1].region
            )
        )
        num_write_vars = len(
            collect_block_iter_vars_used_in_access_region(
                block_stmt, block_stmt.writes[0].region
            )
        )
        if num_read_vars0 == num_read_vars1 == num_write_vars:
            return True
        return False

    def __str__(self) -> str:
        return f'BlockInfo("{self.name}", "{self.dom_kind()}", {self.dom()})'

    def __repr__(self) -> str:
        return str(self)


class FuncInfo:
    """Information about a TIR prim function
    """    
    name: str
    func: tir.PrimFunc
    sch: tir.Schedule
    block_infos: List[BlockInfo]
    kind: FuncKind
    dynamic_args: Optional[List[tir.Var]]

    def __init__(self, func: tir.PrimFunc, name: str = TVM_DEFAULT_NAME):
        """Construct a FuncInfo object."""
        self.name = name
        self.func = func
        self.sch = self.create_schedule()
        self.block_infos, self.has_reduction, self.first_reduction_block = normalize_prim_func(self.sch)
        self.kind = self.get_func_kind()
        self.args = [buffer for buffer in func.buffer_map.values()]
        self.use_fp16 = any([x.dtype == "float16" for x in self.args])
        self.dynamic_args = self.get_dynamic_args()

    def create_schedule(self) -> tir.Schedule:
        return tir.Schedule(self.func)

    def get_dynamic_args(self) -> Optional[List[tir.Var]]:
        dynamic_args = []
        for buffer in self.func.buffer_map.values():
            for dim in buffer.shape:
                if not isinstance(dim, tir.expr.IntImm) and dim not in dynamic_args:
                    if len(dynamic_args) < 2:
                        dynamic_args.append(dim)
                    else:
                        ##TODO change it to error
                        print(
                            "currently only support at most two dynamic shape dimensions"
                        )
        return dynamic_args

    def get_func_kind(self) -> FuncKind:
        """The kind of the function."""
        if get_log_level() >= 2:
            debug_info("Begin to normalize & set FuncKind.......")
        block_infos = self.block_infos
        if not block_infos:
            return FuncKind.KFunc_Fallback
        if self.has_reduction:
            func_kind = self.first_reduction_block.get_reduction_type()
            if func_kind == FuncKind.kFunc_GEMM:
                if get_log_level() >= 1:
                    debug_info("FuncKind == kFunc_GEMM, begin to normalize...")
                (
                    self.gemm_extent_map,
                    self.in_dtype,
                    self.out_dtype,
                    self.t_a,
                    self.t_b,
                    self.wmma_shape_candidats,
                ) = normalize_gemm(self.sch, self.first_reduction_block)
                self.func = self.sch.mod["main"]
                return func_kind
            elif func_kind == FuncKind.kFunc_GEMV:
                is_inner_reduction = normalize_gemv(self.sch, block_infos)
                if is_inner_reduction == None:
                    return FuncKind.KFunc_Fallback
                else:
                    self.func = self.sch.mod["main"]
                    if get_log_level() >= 1:
                        debug_info(
                            f"FuncKind == kFunc_GEMV ,inner_reduction = {is_inner_reduction}"
                        )
                return func_kind
            else:
                if get_log_level() >= 1:debug_info("FuncKind == kFunc_Reduction,begin to normalize..." )
                self.general_red, self.block_infos, self.dyn_red, self.red_len, self.in_dtype = is_general(self.sch, block_infos)
                if self.general_red:
                    if get_log_level() >= 1: debug_info("is_general_reduction")
                    self.num_leading_s, self.num_trailing_r, self.rows = normalize_general_reduction(self.sch, self.block_infos)
                else:
                    if get_log_level() >= 1:
                        debug_info("not_general_reduction")
                    (
                        self.is_inner_reduction,
                        self.c_factor,
                        self.loop_order,
                        self.s_split_index,
                    ) = normalize_reduction(self.sch, self.blocks)
                self.func = self.sch.mod["main"]
                # debug_info(self.sch.mod)
                # if self.dyn_red:
                #     assert_stmt = tir.AssertStmt(self.red_len > 0, tvm.runtime.String(f"{self.red_len} should be greater than 0"), self.func.body.block.body)
                #     old_block = self.func.body.block
                #     new_block = tir.Block(old_block.iter_vars, old_block.reads, 
                #                           old_block.writes, old_block.name_hint, 
                #                           assert_stmt, old_block.init, old_block.alloc_buffers, 
                #                           old_block.match_buffers, old_block.annotations)
                #     new_func_body = tir.BlockRealize(self.func.body.iter_values, self.func.body.predicate, new_block)
                #     self.func = self.func.with_body(new_func_body)
                    # debug_info(self.func.body.block.body)
                    # debug_info(new_block)
                    # debug_info(f"type:\n{type(self.func)}")
                    # debug_info(f"type:\n{type(self.func.body)}")
                    # debug_info(f"type:\n{type(self.func.body.block)}")
                    # debug_info(f"new_block_body:\n{self.func}")
                # debug_info(self.func)
                return FuncKind.kFunc_Reduction
        elif blocks[0].is_elementwise(self.sch):
            if get_log_level() >= 2:
                debug_info(
                    "FuncKind == kFunc_Elementwise, begin to normalize......."
                )
            self.in_dtype, self.num_elements = normalize_elementwise(
                self.sch, blocks
            )
            self.func = self.sch.mod["main"]
            # print(self.func)
            return FuncKind.kFunc_Elementwise
        elif blocks[0].is_transpose():
            self.func = self.sch.mod["main"]
            return FuncKind.kFunc_Transpose
        return FuncKind.KFunc_Fallback

    def __str__(self) -> str:
        return f'FuncInfo("{self.name}", {self.blocks})'

    def __repr__(self) -> str:
        return str(self)


_normalize_prim_func = get_global_func("tir.schedule.NormalizePrimFunc")


def normalize_prim_func(sch: tir.Schedule) -> Optional[Tuple[List[BlockInfo], bool, BlockInfo]]:
    """Normalize the primfunc to normal form
    """    
    try:
        result = _normalize_prim_func(sch)
        if result is None:
            print("result is None")
            return None
    except Exception:  # pylint: disable=broad-except
        print("exception")
        return None

    def _iter_kind(i: tir.IterVar) -> str:
        return {
            tir.IterVar.DataPar: "S",
            tir.IterVar.CommReduce: "R",
        }.get(i.iter_type, "O")
    blocks: List[BlockInfo] = []
    has_reduction = False
    first_reduction_block = None
    for block, loops, iters, is_reduction in zip(*result):
        blocks.append(
            BlockInfo(
                name=sch.get(block).name_hint,
                sch=sch,
                iters=[
                    IterInfo(
                        kind=_iter_kind(iter),  # type: ignore
                        var=iter.var,
                        dom=iter.dom.extent,
                        loop_rv=loop,
                    )
                    for loop, iter in zip(loops, iters)
                ],
                block_rv=block,
                reduction_block=is_reduction,
            )
        )
        if not has_reduction:
            if is_reduction:
                has_reduction = is_reduction
                first_reduction_block = blocks[-1]
    return (blocks, has_reduction, first_reduction_block)


def can_apply_wmma(wmma_shape, in_dtype, out_dtype) -> bool:
    if wmma_shape in [(16, 16, 16), (8, 32, 16), (32, 8, 16)]:
        if in_dtype not in ["float16", "int8"]:
            return False
        if in_dtype == "float16" and out_dtype not in ["float16", "float32"]:
            return False
        if in_dtype == "int8" and out_dtype != "int32":
            return False
        return True
    if wmma_shape == (8, 8, 32):
        if in_dtype != "int4" or out_dtype != "int32":
            return False
        return True


def normalize_gemm(sch: tir.Schedule, first_reduction_block:BlockInfo):
    """    Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]

    Args:
        sch (tir.Schedule): the schedule to be transformed
        first_reduction_block (BlockInfo): first_reduction_block

    Returns:
    """    
    # block_infos = normalize_prim_func(sch)
    ## TODO 报错管理
    root_block = get_root_block(sch)
    blocks = sch.get_child_blocks(root_block)
    # We first inline all transpose blocks for later analysis of transposed A and B
    blocks = inline_transpose_block(sch, blocks)
    
    main_block = first_reduction_block.block_rv
    block_stmt = sch.get(main_block)
    in_dtype, out_dtype = (
        block_stmt.reads[0].buffer.dtype,
        block_stmt.writes[0].buffer.dtype,
    )
    index_maps = get_index_map(block_stmt)
    if index_maps is None:
        return None
    matmul_index_map, a_index_map, b_index_map, c_index_map = index_maps
    gemm_extent_map = {}
    # Step 0. Normalize generic matmul to C[S, I, J] += A[S, I, K] * B[S, J, K]
    block = sch.reindex(main_block, ("read", 0))
    sch.transform_layout(block, ("write", 0), a_index_map)
    is_transpose_a = is_transpose_block(sch.get(block))
    block_stmt = sch.get(block)
    b, m, k = [r.min for r in block_stmt.writes[0].region]
    for _iter in block_stmt.iter_vars:
        if isinstance(b, tir.Var):
            if _iter.var.name == b.name:
                gemm_extent_map["s"] = _iter.dom.extent
        else:
            # the batch dim 直接变成read[0]了
            gemm_extent_map["s"] = tir.IntImm("int32", 1)
        if _iter.var.name == m.name:
            gemm_extent_map["m"] = _iter.dom.extent
        if _iter.var.name == k.name:
            gemm_extent_map["k"] = _iter.dom.extent
    block = sch.reindex(main_block, ("read", 1))
    sch.transform_layout(block, ("write", 0), b_index_map)
    is_transpose_b = is_identity_block(sch.get(block))
    
    block_stmt = sch.get(block)
    _, n, _ = [r.min for r in block_stmt.writes[0].region]
    for _iter in block_stmt.iter_vars:
        if _iter.var.name == n.name:
            gemm_extent_map["n"] = _iter.dom.extent
    block = sch.reindex(main_block, ("write", 0))
    sch.transform_layout(block, ("read", 0), c_index_map)
    sch.transform_block_layout(main_block, matmul_index_map)
    
    minimal_tensorize_threshold = 64
    apply_tensorization: bool = True
    # the batch dimension is not taken into consideration.
    for item_var in block_stmt.iter_vars[1:]:
        extent = item_var.dom.extent
        if isinstance(extent, tir.expr.IntImm):
            if extent.value <= minimal_tensorize_threshold:
                apply_tensorization = False
    wmma_shape_lists = [(16, 16, 16), (8, 32, 16), (32, 8, 16), (8, 8, 32)]
    wmma_shape_candidates = []
    if apply_tensorization:
        for wmma_shape in wmma_shape_lists:
            if can_apply_wmma(wmma_shape, in_dtype, out_dtype):
                wmma_shape_candidates.append(wmma_shape)
    # print("line364 analisys.py", gemm_extent_map)
    return gemm_extent_map, in_dtype, out_dtype, is_transpose_a, is_transpose_b, wmma_shape_candidates


def normalize_gemv(sch: tir.Schedule, block_infos: List[BlockInfo]):
    """Normalize the main block."""

    block_infos = try_inline_contiguous_spatial(sch, block_infos)
    block_info = block_infos[0]
    block_stmt: tir.Block = sch.get(block_info.block_rv)

    access = arith.normalize_to_iter_sum(
        detect_dominant_read(block_stmt),
        input_iters={i.var: i.dom for i in block_stmt.iter_vars},
    )
    # print(access.base)
    buffers_use_vars = [
        collect_block_iter_vars_used_in_access_region(block_stmt, buf.region)
        for buf in block_stmt.writes
    ]
    buffers_use_vars.extend(
        [
            collect_block_iter_vars_used_in_access_region(block_stmt, buf.region)
            for buf in block_stmt.reads
        ]
    )
    if access.base != 0:
        return None
    iter_to_info = {i.var: i for i in block_info.iters}
    batch_loops, s_loops, r_loops, c_loops = [], [], [], []
    inner_axis = access.args[-1].source.source
    # for arg in access.args:
    #     print(arg)
    is_inner_reduction = iter_to_info[inner_axis].kind == "R"
    # print("zhe",is_inner_reduction)
    for split_expr in access.args:
        var = split_expr.source.source
        # print(var)
        info = iter_to_info.get(var)
        loop = info.loop_rv
        is_reduction = info.kind == "R"
        if split_expr.lower_factor > 1:
            if c_loops:
                return None
            loop, c_loop = sch.split(loop, factors=[None, split_expr.lower_factor])
            # we only support the reduction dim being grouped atm
            if not is_reduction:
                return None
            c_loops.append(c_loop)
        if is_reduction:
            r_loops.append(loop)
        elif all([var in buf_vars for buf_vars in buffers_use_vars]):
            batch_loops.append(loop)
        else:
            # print("here")
            s_loops.append(loop)

    assert s_loops
    assert r_loops
    if not c_loops:
        c_loops = [sch.add_unit_loop(block_info.block_rv)]
    if not batch_loops:
        batch_loops = [sch.add_unit_loop(block_info.block_rv)]
    sch.reorder(*batch_loops, *s_loops, *r_loops, *c_loops)
    sch.fuse(*batch_loops)
    sch.fuse(*s_loops)
    sch.fuse(*r_loops)
    # print(sch.mod)
    return is_inner_reduction


def is_general(sch: tir.Schedule, block_infos: List[BlockInfo]):
    ## TODO 报错管理
    # debug_info(sch.mod)
    block_infos = try_inline_contiguous_spatial(sch, block_infos)
    is_general = False
    in_dtype = sch.get(block_infos[0].block_rv).reads[0].buffer.dtype
    block_stmt = sch.get(block_infos[0].block_rv)
    is_dynamic_reduction = False
    if not isinstance(block_stmt.iter_vars[-1].dom.extent, tir.IntImm):
        is_dynamic_reduction = True
        reduction_len = block_stmt.iter_vars[-1].dom.extent
    else:
        reduction_len = block_stmt.iter_vars[-1].dom.extent.value
    if len(block_infos) > 2 or len(block_stmt.writes) != 1:
        # softmax len(block_infos) > 2, layernorm  len(block_stmt.writes) ==2
        is_general = True
    # debug_info(sch.mod)
    return is_general, block_infos, is_dynamic_reduction, reduction_len, in_dtype


def normalize_reduction(  # pylint: disable=too-many-branches
    sch: tir.Schedule,
    block_info: BlockInfo,
) -> Tuple[Optional[bool], Optional[int]]:
    block_stmt = sch.get(block_info.block_rv)
    access: arith.IterSumExpr = arith.normalize_to_iter_sum(
        detect_dominant_read(block_stmt),
        input_iters={i.var: i.dom for i in block_stmt.iter_vars},
    )
    if access.base != 0:
        return None, None
    iter_to_info = {i.var: i for i in block_info.iters}
    s_loops, r_loops, c_loops, c_factor = [], [], [], None
    s_split_loop, s_split_index = None, None
    for split_expr in access.args:
        var = split_expr.source.source
        info = iter_to_info.pop(var)
        loop = info.loop_rv
        is_inner_reduction = info.kind == "R"
        if split_expr.lower_factor > 1:
            if c_loops:
                return None, None
            s_split_loop = loop
            s_split_index = len(s_loops)
            loop, c_loop = sch.split(loop, factors=[None, split_expr.lower_factor])
            c_loops.append(c_loop)
            if not is_inner_reduction:
                c_factor = split_expr.lower_factor
        if is_inner_reduction:
            r_loops.append(loop)
        else:
            s_loops.append(loop)

    if iter_to_info:
        for var, info in iter_to_info.items():
            if info.kind == "S" and info.dom == 1:
                s_loops.append(info.loop_rv)
            else:
                return None, None

    loop_order = {}
    s_block_var_loops = []
    for i in block_info.iters:
        if i.loop_rv in s_loops or i.loop_rv == s_split_loop:
            s_block_var_loops.append(i.loop_rv)

    for i in range(len(s_block_var_loops)):
        for j in range(len(s_loops)):
            if s_block_var_loops[i] == s_loops[j]:
                loop_order[i] = j
                break
            if s_block_var_loops[i] == s_split_loop:
                loop_order[i] = s_split_index
                break

    assert s_loops
    assert r_loops
    if len(s_loops) != len([i for i in block_info.iters if i.kind == "S"]):
        return None, None
    if not c_loops:
        c_loops = [sch.add_unit_loop(block_info.block_rv)]
    sch.reorder(*s_loops, *r_loops, *c_loops)
    sch.fuse(*s_loops)
    sch.fuse(*r_loops)
    return is_inner_reduction, c_factor, loop_order, s_split_index


def normalize_general_reduction(sch: tir.Schedule, block_infos: List[BlockInfo]):
    """Normalize the main block."""
    dom_kind = block_infos[0].dom_kind()
    num_leading_s = len(dom_kind) - len(dom_kind.lstrip("S"))
    num_trailing_r = len(dom_kind) - len(dom_kind.rstrip("R"))
    # Align the number of block iters of the last block.
    num_last_block_iter = len(block_infos[-1].dom_kind())
    if num_last_block_iter < len(dom_kind):
        index_map = tir.IndexMap.from_func(
            lambda *iters: (
                [tir.const(0, iters[0].dtype)] * (len(dom_kind) - num_last_block_iter)
                + list(iters)
            ),
            ndim=num_last_block_iter,
        )
        sch.transform_block_layout(block_infos[-1].block_rv, index_map)
    loops = sch.get_loops(block_infos[-1].block_rv)
    rows = 1
    for loop in loops[: num_leading_s]:
        if isinstance(sch.get(loop).extent, tir.IntImm):
            rows *= sch.get(loop).extent.value
    # rows = sch.get(sch.fuse(*loops[: num_leading_s])).extent.value
    debug_info(f"rows:{rows}")
    # debug_info(f"num_leading_s:{num_leading_s},num_trailing_r:{num_trailing_r}")
    try:
    # TODO: fix num_leading_s = 0 case
        assert num_trailing_r > 0, "num_trailing_r > 0"
        for block in block_infos[1:-1]:
            assert block.dom_kind() == dom_kind, f"block.dom_kind: {block.dom_kind()}, {dom_kind}"
        assert block_infos[-1].is_injective(),"block_infos[-1].is_injective()"
        assert len(block_infos[-1].dom_kind()) <= len(dom_kind),"len(block_infos[-1].dom_kind()) <= len(dom_kind)"
    except AssertionError:
        debug_info(f"Failed to normalize general reduction: {dom_kind}")
        return None
    # debug_info(sch.mod)
    return num_leading_s, num_trailing_r, rows


def normalize_elementwise(sch: tir.Schedule, block_infos: List[BlockInfo]):
    block_infos = try_inline_contiguous_spatial(sch, block_infos)
    block = block_infos[0].block_rv
    loops = sch.get_loops(block)
    sch.fuse(*loops)
    block_stmt = sch.get(block)
    in_dtype = block_stmt.reads[0].buffer.dtype
    num_elements = 1
    for i in block_stmt.iter_vars:
        if isinstance(i.dom.extent, tir.IntImm):
            num_elements *= i.dom.extent.value
    if get_log_level() >= 2:
        debug_info(f"num_elements: {num_elements}, in_dtype: {in_dtype}")
    return in_dtype, num_elements


def make_iter_fusion_index_map(
    traits: List[IterTrait],
    kind_order: List[IterKind],
) -> tir.IndexMap:
    fused_iters: Dict[IterKind, PrimExpr] = {}
    input_iters: List[tir.Var] = []
    for i, trait in enumerate(traits):
        v_i = tir.Var(f"i{i}", trait.extent.dtype)
        input_iters.append(v_i)
        if trait.kind == IterKind.kIter_T:
            continue
        if trait.kind not in kind_order:
            raise ValueError(f"Unknown iter kind {trait.kind}")
        if trait.kind in fused_iters:
            fused_iters[trait.kind] = fused_iters[trait.kind] * trait.extent + v_i
        else:
            fused_iters[trait.kind] = v_i

    final_indices: List[tir.PrimExpr] = [
        fused_iters.get(kind, tir.IntImm(traits[0].extent.dtype, 0))
        for kind in kind_order
    ]

    return tir.IndexMap(input_iters, final_indices, None)


def detect_iter_traits(block: tir.Block) -> Optional[Tuple[List[IterTrait]]]:
    """Detect iter traits based on the pattern C[S, I, J] += A[S, I, K] * B[S, J, K]

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    simplified : bool
        Whether to use simplified index map (e.g. remove constant axes)

    Returns
    -------
    traits : Optional[Tuple[List[IterTrait]]]
        The detected iter traits for axes in A, B and C. None if the block
        does not match the pattern.

    """

    if len(block.reads) != 2 or len(block.writes) != 1:
        return None

    def get_access_axes(region: List[Range]) -> Set[Var]:
        axes: Set[Var] = set()
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes = axes.union(set(undefined_vars(r.min)))
        return axes

    try:
        A_axes = get_access_axes(block.reads[0].region)
        B_axes = get_access_axes(block.reads[1].region)
        C_axes = get_access_axes(block.writes[0].region)
    except ValueError:
        return None

    traits: Dict[Var, IterTrait] = {}
    for iter_var in block.iter_vars:
        var = iter_var.var
        kind: IterKind
        if _is_one(iter_var.dom.extent):
            if iter_var.iter_type == tir.IterVar.CommReduce:
                # for simplified case (e.g. 1x1 conv kernel)
                kind = IterKind.kIter_K
            else:
                kind = IterKind.kIter_T
        elif iter_var.iter_type == iter_var.DataPar:
            if var in A_axes and var in B_axes and var in C_axes:
                kind = IterKind.kIter_S
            elif var in A_axes and var in C_axes:
                kind = IterKind.kIter_I
            elif var in B_axes and var in C_axes:
                kind = IterKind.kIter_J
            else:
                return None
        elif iter_var.iter_type == tir.IterVar.CommReduce:
            if var in A_axes and var in B_axes and var not in C_axes:
                kind = IterKind.kIter_K
            else:
                return None
        else:
            return None
        traits[var] = IterTrait(kind, iter_var.dom.extent)

    # A Gemm-kernel requires have I, J and K axes
    gemm_traits = {IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K}
    if {x.kind for x in traits.values()}.intersection(gemm_traits) != gemm_traits:
        return None

    A_traits = [
        traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in A_axes
    ]
    B_traits = [
        traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in B_axes
    ]
    C_traits = [
        traits[iter_var.var] for iter_var in block.iter_vars if iter_var.var in C_axes
    ]
    block_traits = [traits[i.var] for i in block.iter_vars]
    return A_traits, B_traits, C_traits, block_traits


def get_index_map(
    block: tir.Block, layout: List[str] = ["n", "t", "n"]
) -> Optional[Tuple[tir.IndexMap, ...]]:
    """Get index maps for the block

    Parameters
    ----------
    block : tir.Block
        The block to be analyzed

    layout : List[str]
        the target layout index map to be used.
        'n' for [i, k] layout
        't' for [k, j] layout
        'a' for auto inference based on whether the last axis is reduction.

    Returns
    -------
    index_maps : Optional[Tuple[tir.IndexMap]]
        The index maps for the block, or None if the block is not a gemm-liked kernel
    """
    traits = detect_iter_traits(block)
    if traits is None:
        return None
    A_traits, B_traits, C_traits, block_traits = traits

    def get_ordered_axes(region: List[Range]) -> Set[Var]:
        axes: List[Var] = []
        for r in region:
            if not _is_one(r.extent):
                raise ValueError("Expect elemwise block access")
            axes.append(r.min)
        return axes

    def is_common_reduce(var: Var) -> bool:
        for iter_var in block.iter_vars:
            if iter_var.var == var and iter_var.iter_type == IterVar.CommReduce:
                return True
        return False

    def check_last_trait(region: List[Range]):
        axes = get_ordered_axes(region)
        return is_common_reduce(axes[-1])

    def infer_layout(layout: str, region: List[Range], kind: str = "A"):
        """
        Infer the layout based on the region and the kind of buffer
        kind: "A", "B", "C"
        """
        primary_iter, secondary_iter, reduction_iter = {
            "A": (IterKind.kIter_I, IterKind.kIter_K, IterKind.kIter_K),
            "B": (IterKind.kIter_K, IterKind.kIter_J, IterKind.kIter_K),
            "C": (IterKind.kIter_I, IterKind.kIter_J, None),
        }[kind]

        spatial_iter = {
            "A": IterKind.kIter_I,
            "B": IterKind.kIter_J,
            "C": None,
        }[kind]

        if layout == "n":
            return [IterKind.kIter_S, primary_iter, secondary_iter]
        elif layout == "t":
            return [IterKind.kIter_S, secondary_iter, primary_iter]
        elif layout == "a":
            # auto inference layout
            # for buffer with reduction axis, we put it as the last axis
            # otherwise, we put it as the first axis
            if kind == "C":
                return [IterKind.kIter_S, primary_iter, secondary_iter]
            else:
                return (
                    [IterKind.kIter_S, spatial_iter, reduction_iter]
                    if check_last_trait(region)
                    else [IterKind.kIter_S, reduction_iter, spatial_iter]
                )
        else:
            raise ValueError(f"Unknown layout {layout}")

    A_index_map = make_iter_fusion_index_map(
        A_traits, infer_layout(layout[0], block.reads[0].region, kind="A")
    )
    B_index_map = make_iter_fusion_index_map(
        B_traits, infer_layout(layout[1], block.reads[1].region, kind="B")
    )
    C_index_map = make_iter_fusion_index_map(
        C_traits, infer_layout(layout[2], block.writes[0].region, kind="C")
    )

    matmul_index_map = make_iter_fusion_index_map(
        block_traits, [IterKind.kIter_S, IterKind.kIter_I, IterKind.kIter_J, IterKind.kIter_K]
    )

    return (
        matmul_index_map,
        A_index_map,
        B_index_map,
        C_index_map,
    )


def try_inline(
    sch: tir.Schedule,
    blocks: List[BlockInfo],
) -> List[BlockInfo]:
    """Try to inline as many blocks as possible, and return the remaining blocks.

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to inline blocks.
    blocks : List[BlockInfo]
        The blocks to be inlined.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    def _trial(func: Callable):
        for i, block in enumerate(blocks):
            try:
                func(block.block_rv)
            except:  # pylint: disable=bare-except
                continue
            return i
        return None

    while True:
        i = _trial(sch.compute_inline)
        if i is None:
            i = _trial(sch.reverse_compute_inline)
        if i is None:
            break
        blocks.pop(i)
    return blocks


def try_inline_contiguous_spatial(
    sch: tir.Schedule,
    block_infos: List[BlockInfo],
) -> List[BlockInfo]:
    """Try to inline contiguous spatial blocks in a schedule

    Parameters
    ----------
    sch : tir.Schedule
        The TIR schedule used to inline blocks.
    block_infos : List[BlockInfo]
        The blocks to be try.

    Returns
    -------
    remaining : List[BlockInfo]
        The remaining blocks that cannot be inlined.
    """

    if block_infos is None:
        return None
    results = []
    spatial_blocks = []
    block: BlockInfo
    for block in block_infos:
        if block.is_injective():
            spatial_blocks.append(block)
        elif spatial_blocks:
            results.extend(try_inline(sch, spatial_blocks))
            results.append(block)
            spatial_blocks = []
        else:
            results.append(block)
    if spatial_blocks:
        results.extend(try_inline(sch, spatial_blocks))
    return results
