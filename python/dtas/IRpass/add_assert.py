from typing import List, Optional

import tvm
from tvm import tir
from tvm.ir import IRModule
from tvm.ir.transform import PassContext, module_pass

@module_pass(opt_level=0, name="AddAssert")
class AddAssert:  # pylint: disable=too-few-public-methods
    """A IRModule pass that add assert that dynamic shape > 0 if there are dynamic args to all PrimFuncs in the module."""

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
