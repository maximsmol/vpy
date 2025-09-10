from ctypes import CFUNCTYPE, c_int64

import llvmlite.binding as llvm

from vpy.compile import Compiler

from .lex import Lexer
from .parse import Parser


def setup_llvm() -> llvm.ExecutionEngine:
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()

    backing_mod = llvm.parse_assembly("")
    return llvm.create_mcjit_compiler(backing_mod, target_machine)


def execute(engine: llvm.ExecutionEngine, code: str) -> int:
    mod = llvm.parse_assembly(code)
    mod.verify()

    engine.add_module(mod)
    try:
        engine.finalize_object()
        engine.run_static_constructors()

        func_ptr = engine.get_function_address("test")
        cfunc = CFUNCTYPE(c_int64)(func_ptr)
        return cfunc()
    finally:
        engine.remove_module(mod)


def main() -> None:
    engine = setup_llvm()

    # src = "1 + 2 + 3"
    # src = "a = 123\na + 10"
    # src = "10 % 6"
    # src = "1 == 1"
    # src = "2 * 5"
    # src = "a = 1\na += 2\na"
    src = "True"

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast = p.parse()

    c = Compiler()
    c.compile(ast)

    llvm_ir = "\n".join(c.lines)
    print(llvm_ir)

    res = execute(engine, llvm_ir)

    print()
    print("test() =", res)

    cur = p.tok()
    if cur.type != "endmarker":
        print()
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()


if __name__ == "__main__":
    main()
