import ctypes
from ctypes import (
    CFUNCTYPE,
    POINTER,
    Structure,
    c_bool,
    c_double,
    c_int64,
    c_uint8,
    c_uint64,
    cast,
    pointer,
)
from pathlib import Path
from textwrap import dedent

import llvmlite

from vpy.compile import Compiler, Scope

from .lex import Lexer
from .parse import Parser

# https://llvmlite.readthedocs.io/en/stable/user-guide/deprecation.html#deprecation-of-typed-pointers
llvmlite.opaque_pointers_enabled = True

import llvmlite.binding as llvm


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

        func_ptr = engine.get_function_address("module_root")
        cfunc = CFUNCTYPE(c_int64)(func_ptr)
        return cfunc()
    finally:
        engine.remove_module(mod)


class CompilerValue(Structure):
    _fields_ = [("type_id", c_uint64), ("data", POINTER(c_uint8))]


class CompilerValueDataInt(Structure):
    _fields_ = [("value", c_int64)]


class CompilerValueDataBool(Structure):
    _fields_ = [("value", c_bool)]


class CompilerValueDataFloat(Structure):
    _fields_ = [("value", c_double)]


def to_python(ptr: int) -> object:
    val = CompilerValue.from_address(ptr)
    match val.type_id:
        case 0:
            return None
        case 1:
            data = cast(pointer(val.data), POINTER(CompilerValueDataInt)).contents
            return data.value
        case 2:
            data = cast(pointer(val.data), POINTER(CompilerValueDataBool)).contents
            return data.value
        case 3:
            data = cast(pointer(val.data), POINTER(CompilerValueDataFloat)).contents
            return data.value
        case x:
            raise RuntimeError(f"unknown compiler value type: {x}")


def main() -> None:
    engine = setup_llvm()

    # src = "1 + 2 + 3"
    # src = "a = 123"
    # src = dedent("""
    #     a = 123
    #     a + 1
    # """)[1:]
    # src = "10 % 6"
    # src = "1 == 1"
    # src = "2 * 5"
    # src = dedent("""
    #     a = 1
    #     a += 2
    # """)[1:]
    # src = "True"
    # src = dedent("""
    #     a = 10
    #     if False:
    #         a = 20
    #     a
    # """)[1:]
    # src = dedent("""
    #     a = 2
    #     while a <= 10:
    #         a = a * a
    #     a
    # """)[1:]
    # src = dedent("""
    #     def f(x: int) -> int:
    #         return x + 10

    #     f(10)
    # """)[1:]
    src = dedent("""
        def f(x: int) -> int | float:
            if x == 5:
                return 123
            return 0.999

        f(10)
    """)[1:]
    # src = Path("tests/problems_99/p2.01_is_prime.py").read_text(encoding="utf-8")
    # src = Path("tests/problems_99/p2.07_gcd.py").read_text(encoding="utf-8")

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast = p.parse()

    c = Compiler()

    root = Scope(compiler=c)
    c.scopes.append(root)

    root.compile(ast)

    llvm_ir = "\n\n".join("\n".join(s.lines) for s in c.scopes)
    print(llvm_ir)

    res = execute(engine, llvm_ir)

    print()
    print(f"module_root() = 0x{res:x}")
    print(to_python(res))

    cur = p.tok()
    if cur.type != "endmarker":
        print()
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()


if __name__ == "__main__":
    main()
