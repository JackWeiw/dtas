from typing import List, Set, Optional
import numpy as np
import os

from tvm import ir, tir
from tvm.ir import Range
from tvm.tir import IterVar, PrimExpr, Var
from tvm.tir.schedule import BlockRV
from tvm.tir.analysis import undefined_vars

def _collect_producers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for producer in sch.get_producers(block):
        result.append(producer)
        result.extend(_collect_producers(sch, producer))
    return result


def _collect_consumers(sch: tir.Schedule, block: tir.schedule.BlockRV):
    result = []
    for consumer in sch.get_consumers(block):
        result.append(consumer)
        result.extend(_collect_consumers(sch, consumer))
    return result


def auto_inline_producers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
    skip_blocks: Optional[List[tir.schedule.BlockRV]] = None,
):
    skip_blocks = skip_blocks or []
    while True:
        inlined_cnt = 0
        producers = _collect_producers(sch, block)
        for producer in producers:
            if any(
                sch.get(producer) == sch.get(skip_block) for skip_block in skip_blocks
            ):
                continue
            try:
                sch.compute_inline(producer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumers(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    while True:
        inlined_cnt = 0
        consumers = _collect_consumers(sch, block)
        for consumer in consumers:
            try:
                sch.compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        for consumer in consumers:
            try:
                sch.reverse_compute_inline(consumer)
                inlined_cnt += 1
            except:  # pylint: disable=bare-except
                continue
        if inlined_cnt == 0:
            return


def auto_inline_consumer_chain(
    sch: tir.Schedule,
    block: tir.schedule.BlockRV,
):
    auto_inline_consumers(sch, block)
    remaining_consumers = sch.get_consumers(block)

    if len(remaining_consumers) != 0:
        # Some blocks have failed to be inlined to the producer cache-write stage.
        # This could be due to another producer block that has not been scheduled.
        for c in remaining_consumers:
            for p in sch.get_producers(c):
                if sch.get(p) != sch.get(block):
                    auto_inline_producers(sch, p)
                    sch.compute_inline(p)

        # Try inlining into the cache-write stage again, this time it should succeed.
        auto_inline_consumers(sch, block)

    msg = "There are some consumers of the cache-write stage that are not properly inlined."
    assert len(sch.get_consumers(block)) == 0, msg


def get_all_factors(n: int) -> List[int]:
    n0 = int(np.ceil(np.sqrt(n)))
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]


def factorize(n: int) -> List[int]:
    i = 2
    result = []
    while n > 1:
        if n % i == 0:
            n //= i
            result.append(i)
        else:
            i += 1
    return result


def get_max_factor(n, factors):
    factors = sorted(factors, reverse=True)
    for factor in factors:
        if n % factor == 0:
            return factor
    return 1


def save_to_file(filename, sch, force_overwrite=False):
    if force_overwrite:
        with open(filename, "w") as f:
            f.write(sch.mod.script())
    else:
        if not os.path.exists(filename) or os.stat(filename).st_size == 0:
            print("save to file: ", filename)
            with open(filename, "w") as f:
                f.write(sch.mod.script())
        else:
            print("fail to save to file: ", filename, "file is not empty")


def collect_block_iter_vars_used_in_access_region(
    block: tir.Block, region: List[ir.Range]
) -> Set[tir.Var]:
    """Collect the block iter variables used in the access region of a buffer region."""
    tir_vars = set()
    for expr in region:
        assert expr.extent == 1
        tir_vars |= collect_vars_used_in_prim_expr(expr.min)
    tir_vars &= set(iter_var.var for iter_var in block.iter_vars)
    return tir_vars


def collect_vars_used_in_prim_expr(expr: tir.PrimExpr) -> Set[tir.Var]:
    """Collect the variables used in the PrimExpr."""
    tir_vars = set()

    def _collect_tir_var(expr):
        if isinstance(expr, tir.Var):
            tir_vars.add(expr)

    tir.stmt_functor.post_order_visit(expr, _collect_tir_var)
    return tir_vars


def detect_dominant_read(block: tir.Block) -> tir.PrimExpr:
    """Detect the dominant read indices in the block."""
    dominant_read = None
    num_read_iters = -1
    for buffer_region in block.reads:
        tir_vars = collect_block_iter_vars_used_in_access_region(
            block, buffer_region.region
        )
        if num_read_iters < len(tir_vars):
            num_read_iters = len(tir_vars)
            dominant_read = buffer_region
    assert dominant_read is not None
    (result,) = dominant_read.buffer.offset_of([e.min for e in dominant_read.region])
    return result


def get_root_block(sch: tir.Schedule, func_name: str = "main") -> BlockRV:
    try:
        block = sch.mod[func_name].body.block
    except:
        raise ValueError(
            f"The function body is expected to be the root block, but got:\n"
            f"{sch.mod[func_name].body}"
        )
    return sch.get_block(block.name_hint)


def get_reduction_blocks(sch, blocks) -> bool:
    # Get the gemm main computation block
    def is_reduction(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.CommReduce, IterVar.DataPar}

    def is_spatial(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
        return iter_types == {IterVar.DataPar}

    # NOTE: We assume there is only one reduction block in the function
    # all blocks are required to be spatial or reduction
    if not all([is_reduction(block) or is_spatial(block) for block in blocks]):
        return None

    # There is only one reduction block
    reduction_blocks = [block for block in blocks if is_reduction(block)]
    if len(reduction_blocks) != 1:
        return None
    return reduction_blocks


def get_dequantize_block(sch, blocks) -> Optional[BlockRV]:
    # check at least two input and one output
    # at lease one input has uint dtype, and the output dtype is float
    def is_dequantize(block: BlockRV) -> bool:
        block_stmt = sch.get(block)
        if len(block_stmt.reads) < 2:
            return False
        has_uint_input = any("uint" in str(region.buffer.dtype) for region in block_stmt.reads)
        if not has_uint_input:
            return False
        if len(block_stmt.writes) != 1 or "float" not in str(block_stmt.writes[0].buffer.dtype):
            return False
        return True

    dequantize_blocks = [block for block in blocks if is_dequantize(block)]
    return dequantize_blocks[0] if len(dequantize_blocks) == 1 else None


def is_identity_or_transpose_block(block_stmt: tir.Block) -> bool:
    iter_types = {iter_var.iter_type for iter_var in block_stmt.iter_vars}
    if iter_types != {IterVar.DataPar}:
        return False, False
    if not isinstance(block_stmt.body, tir.BufferStore):
        return False, False
    if not isinstance(block_stmt.body.value, tir.BufferLoad):
        return False, False

    def get_access_vars(region: List[Range]) -> List[Var]:
        axes: List[Var] = []
        for r in region:
            if not _is_one(r.extent):
                return None
            axes.extend(undefined_vars(r.min))
        # remove trivial axis
        trivial_vars = set(
            iter_var.var for iter_var in block_stmt.iter_vars if _is_one(iter_var.dom.extent)
        )
        axes = [axis for axis in axes if axis not in trivial_vars]
        # remove duplicate axis
        axes = [var for i, var in enumerate(axes) if i == 0 or var != axes[i - 1]]
        return axes

    lhs_access_vars = get_access_vars(block_stmt.reads[0].region)[-2:]
    rhs_access_vars = get_access_vars(block_stmt.writes[0].region)[-2:]
    is_identity = list(lhs_access_vars) == list(rhs_access_vars)
    is_transpose = list(lhs_access_vars) != list(rhs_access_vars) and set(lhs_access_vars) == set(
        rhs_access_vars
    )
    return is_identity, is_transpose


def is_identity_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[0]


def is_transpose_block(block_stmt: tir.Block) -> bool:
    return is_identity_or_transpose_block(block_stmt)[1]


def inline_transpose_block(sch: tir.Schedule, blocks: List[tir.schedule.BlockRV]):
    result_blocks = []
    for block in blocks:
        if not is_transpose_block(sch.get(block)):
            result_blocks.append(block)
            continue
        try:
            sch.compute_inline(block)
        except:
            try:
                sch.reverse_compute_inline(block)
            except:
                result_blocks.append(block)
    return result_blocks


def _is_one(x: PrimExpr) -> bool:
    return isinstance(x, tir.IntImm) and x.value == 1


def is_injective_epilogue(sch, block: BlockRV):
    block_stmt = sch.get(block)
    for iter_var in block_stmt.iter_vars:
        if iter_var.iter_type != tir.IterVar.DataPar:
            return False
    return True


def is_broadcast_epilogue(sch, gemv: BlockRV, epilogue: BlockRV):
    """Check if the epilogue block is a broadcast pattern"""
    write_buffers = {r.buffer for r in sch.get(gemv).writes}
    epilogue_iters = {i.var: i for i in sch.get(epilogue).iter_vars if i.dom != 1}
    for buffer_region in sch.get(epilogue).reads:
        if buffer_region.buffer not in write_buffers:
            continue
        tir_vars = collect_block_iter_vars_used_in_access_region(
            sch.get(epilogue), buffer_region.region
        )
        if len(tir_vars) < len(epilogue_iters):
            return True
    return False


