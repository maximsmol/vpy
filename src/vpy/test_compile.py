from ctypes import CFUNCTYPE, c_int64
from pathlib import Path
from textwrap import dedent

import llvmlite

from .compile import Compiler, Scope, prelude
from .lex import Lexer
from .parse import Parser
from .test_interpret import reference_eval
from .values import VpyValue

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


def process(
    *, engine: llvm.ExecutionEngine, src: str, print_llvm: bool = False
) -> bool:
    res = True

    l = Lexer(data=src)
    p = Parser(lex=l)
    ast = p.parse()

    cur = p.tok()
    if cur.type != "endmarker":
        print("!!! Leftover tokens:")
        while cur.type != "endmarker":
            print(cur)
            cur = p.tok()

        res = False

    c = Compiler()

    root = Scope(compiler=c)
    c.scopes.append(root)

    root.compile(ast)

    llvm_ir = "\n\n".join("\n".join(s.lines) for s in c.scopes)
    if print_llvm:
        print(llvm_ir[len(prelude) + 1 :])
        print()
        print(f"IR size: {len(llvm_ir)}")

    ours = execute(engine, llvm_ir)
    reference = reference_eval(src)

    # todo(maximsmol): check locals/globals/intermediate results
    ours_py = VpyValue.derive_type(0).from_address(ours).to_python()
    if type(ours_py) is not type(reference.value) or ours_py != reference.value:
        print("!!! Mismatch")
        print("Ours:")
        print(ours_py)
        print()
        print("Reference:")
        print(reference.value)
        print()
        res = False

    return res


def main() -> None:
    engine = setup_llvm()

    # todo(maximsmol): support modules that do not end in an expression

    assert process(engine=engine, src="1 + 2 + 3")
    # assert process(engine=engine, src="a = 123")
    assert process(
        engine=engine,
        src=dedent("""
            a = 123
            a + 1
        """)[1:],
    )
    assert process(engine=engine, src="10 % 6")
    assert process(engine=engine, src="1 == 1")
    assert process(engine=engine, src="2 * 5")
    # assert process(
    #     engine=engine,
    #     src=dedent("""
    #         a = 1
    #         a += 2
    #     """)[1:],
    # )
    assert process(engine=engine, src="True")
    assert process(
        engine=engine,
        src=dedent("""
            a = 10
            if False:
                a = 20
            a
        """)[1:],
    )
    assert process(
        engine=engine,
        src=dedent("""
            a = 2
            while a <= 10:
                a = a * a
            a
        """)[1:],
    )
    assert process(
        engine=engine,
        src=dedent("""
            def f(x: int) -> int:
                return x + 10

            f(10)
        """)[1:],
    )
    assert process(
        engine=engine,
        src=dedent("""
            def f(x: int) -> int | float:
                if x == 5:
                    return 123
                return 0.999

            f(10)
        """)[1:],
    )
    assert process(
        engine=engine,
        src=dedent(r"""
            "hello \"quote\" \n world"
        """)[1:],
        print_llvm=True,
    )

    print("Smoketest OK")

    root_p = Path(__file__).parent.parent.parent / "tests/problems_99"

    for f in root_p.iterdir():
        print(f">>> {f.relative_to(root_p)}:")
        ok = process(engine=engine, src=f.read_text(), print_llvm=True)
        if not ok:
            break
        print("  OK")


if __name__ == "__main__":
    main()
