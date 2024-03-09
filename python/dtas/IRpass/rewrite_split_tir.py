"""
TODO : rewrite_split_tir

a tir pass rewrite split

before rewrite:
@T.prim_func(private=True)
def fused_reshape7_split1(lv1782: T.Buffer((1, 1, 7680), "float16"), var_T_split_sections_intermediate: T.Buffer((1, 1, 32, 80), "float16"), var_T_split_sections_intermediate_1: T.Buffer((1, 1, 32, 80), "float16"), var_T_split_sections_intermediate_2: T.Buffer((1, 1, 32, 80), "float16")):
    T.func_attr({"tir.is_scheduled": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0_ax1_fused_0 in T.thread_binding(3, thread="blockIdx.x"):
        for ax0_ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
            with T.block("T_split_sections"):
                v0 = T.axis.spatial(32, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) // 80)
                v1 = T.axis.spatial(80, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) % 80)
                T.where(ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 < 2560)
                T.reads(lv1782[0, 0, v0 * 240 + v1])
                T.writes(var_T_split_sections_intermediate[0, 0, v0, v1])
                var_T_split_sections_intermediate[0, 0, v0, v1] = lv1782[0, 0, v0 * 240 + v1]
    for ax0_ax1_fused_0 in T.thread_binding(3, thread="blockIdx.x"):
        for ax0_ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
            with T.block("T_split_sections_1"):
                v0 = T.axis.spatial(32, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) // 80)
                v1 = T.axis.spatial(80, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) % 80)
                T.where(ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 < 2560)
                T.reads(lv1782[0, 0, v0 * 240 + (v1 + 80)])
                T.writes(var_T_split_sections_intermediate_1[0, 0, v0, v1])
                var_T_split_sections_intermediate_1[0, 0, v0, v1] = lv1782[0, 0, v0 * 240 + (v1 + 80)]
    for ax0_ax1_fused_0 in T.thread_binding(3, thread="blockIdx.x"):
        for ax0_ax1_fused_1 in T.thread_binding(1024, thread="threadIdx.x"):
            with T.block("T_split_sections_2"):
                v0 = T.axis.spatial(32, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) // 80)
                v1 = T.axis.spatial(80, (ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1) % 80)
                T.where(ax0_ax1_fused_0 * 1024 + ax0_ax1_fused_1 < 2560)
                T.reads(lv1782[0, 0, v0 * 240 + (v1 + 160)])
                T.writes(var_T_split_sections_intermediate_2[0, 0, v0, v1])
                var_T_split_sections_intermediate_2[0, 0, v0, v1] = lv1782[0, 0, v0 * 240 + (v1 + 160)]

after rewrite:
@T.prim_func(private=True)
def WT_split(var_A: T.handle, var_T_split_sections: T.handle, var_T_split_sections_1: T.handle, var_T_split_sections_2: T.handle):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), m, T.int64(32), T.int64(240)), "float16")
    T_split_sections = T.match_buffer(var_T_split_sections, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
    T_split_sections_1 = T.match_buffer(var_T_split_sections_1, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
    T_split_sections_2 = T.match_buffer(var_T_split_sections_2, (T.int64(1), m, T.int64(32), T.int64(80)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(80)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_split_sections[v_ax0, v_ax1, v_ax2, v_ax3],T_split_sections_1[v_ax0, v_ax1, v_ax2, v_ax3],T_split_sections_2[v_ax0, v_ax1, v_ax2, v_ax3])
            T_split_sections[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3]
            T_split_sections_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(80)]
            T_split_sections_2[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3 + T.int64(160)] 

"""
from typing import List, Optional

import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass

@module_pass(opt_level=0, name="AddAssert")
class RewriteSplit:  # pylint: disable=too-few-public-methods
    """A IRModule pass that try to rewrite split like primfunc to all PrimFuncs in the module."""

    def __init__(self, dynamic_args: List[tir.PrimExpr]):
        """Construct a new AddAssert pass.

        Parameters
        ----------
        dynamic_args : dynamic_args
            The dynamic_args to be added assertion to all PrimFuncs in the module.
        """
        self.dynamic_args = dynamic_args

    def transform_module(  # pylint: disable=missing-function-docstring
        self,
        mod: IRModule,
        _: PassContext,
    ) -> IRModule:
        if self.dynamic_args is None:
            return mod
        
        updated_functions = {}
        for g_var, func in mod.functions_items():
            if isinstance(func, tir.PrimFunc) :
                assert_stmts = [dynamic_arg > 0 for dynamic_arg in self.dynamic_args]
                assert_stmt = tir.AssertStmt(*assert_stmts, tvm.runtime.String(f"{self.dynamic_args} should be greater than 0"), func.body.block.body)
                old_block = func.body.block
                new_block = tir.Block(old_block.iter_vars, old_block.reads, 
                                        old_block.writes, old_block.name_hint, 
                                        assert_stmt, old_block.init, old_block.alloc_buffers, 
                                        old_block.match_buffers, old_block.annotations)
                new_func_body = tir.BlockRealize(func.body.iter_values, func.body.predicate, new_block)
                new_func = func.with_body(new_func_body)
                updated_functions[g_var] = new_func
                
        for g_var, func in updated_functions.items():
            mod[g_var] = func
            
        return mod
