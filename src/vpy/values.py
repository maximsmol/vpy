from ctypes import (
    POINTER,
    Array,
    Structure,
    c_bool,
    c_double,
    c_int64,
    c_uint8,
    c_uint64,
    cast,
    pointer,
)
from enum import Enum
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .interpret import Interpreter


class VpyValueDataInt(Structure):
    _fields_ = [("value", c_int64)]
    value: int


class VpyValueDataBool(Structure):
    _fields_ = [("value", c_bool)]
    value: bool


class VpyValueDataFloat(Structure):
    _fields_ = [("value", c_double)]
    value: float


class VpyValueDataString(Structure):
    _fields_ = [("len", c_uint64)]
    len: int
    value: Array[c_uint8]

    @classmethod
    def derive_type(cls, l: int) -> type[Self]:
        class Inner(cls):
            _fields_ = [("value", c_uint8 * l)]

        Inner.__name__ = f"{cls.__name__}[{l}]"
        Inner.__qualname__ = Inner.__name__

        return Inner

    @classmethod
    def from_str(cls, x: str) -> Self:
        value = bytearray(x.encode())
        l = len(value)
        return cls.derive_type(l)(len=l, value=(c_uint8 * l).from_buffer(value))


class VpyTypeId(int, Enum):
    none = 0
    int = 1
    bool = 2
    float = 3
    str = 4
    function = 999


class VpyValue(Structure):
    _fields_ = [("type_id", c_uint64)]

    type_id: int
    data: Array[c_uint8]

    @classmethod
    def derive_type(cls, l: int) -> type[Self]:
        class Inner(cls):
            _fields_ = [("data", c_uint8 * l)]

        Inner.__name__ = f"{cls.__name__}[{l}]"
        Inner.__qualname__ = Inner.__name__

        return Inner

    @classmethod
    def from_none(cls) -> Self:
        return cls(type_id=VpyTypeId.none.value)

    @classmethod
    def from_int(cls, x: int) -> Self:
        body = bytearray(VpyValueDataInt(value=x))
        return cls.derive_type(len(body))(
            type_id=VpyTypeId.int.value, data=(c_uint8 * len(body)).from_buffer(body)
        )

    @classmethod
    def from_bool(cls, x: int) -> Self:
        body = bytearray(VpyValueDataBool(value=x))
        return cls.derive_type(len(body))(
            type_id=VpyTypeId.bool.value, data=(c_uint8 * len(body)).from_buffer(body)
        )

    @classmethod
    def from_float(cls, x: float) -> Self:
        body = bytearray(VpyValueDataFloat(value=x))
        return cls.derive_type(len(body))(
            type_id=VpyTypeId.float.value, data=(c_uint8 * len(body)).from_buffer(body)
        )

    @classmethod
    def from_str(cls, x: str) -> Self:
        body = bytearray(VpyValueDataString.from_str(x))
        return cls.derive_type(len(body))(
            type_id=VpyTypeId.str.value, data=(c_uint8 * len(body)).from_buffer(body)
        )

    @classmethod
    def interpreted_from_function(cls, x: str) -> Self:
        body = bytearray(x.encode())
        return cls.derive_type(len(body))(
            type_id=VpyTypeId.function.value,
            data=(c_uint8 * len(body)).from_buffer(body),
        )

    def expect_none(self) -> None:
        assert self.type_id == VpyTypeId.none.value

    def expect_int(self) -> int:
        assert self.type_id == VpyTypeId.int.value

        data = cast(pointer(self.data), POINTER(VpyValueDataInt)).contents
        return data.value

    def expect_bool(self) -> bool:
        assert self.type_id == VpyTypeId.bool.value

        data = cast(pointer(self.data), POINTER(VpyValueDataBool)).contents
        return data.value

    def expect_float(self) -> float:
        assert self.type_id == VpyTypeId.float.value

        data = cast(pointer(self.data), POINTER(VpyValueDataFloat)).contents
        return data.value

    def expect_str(self) -> str:
        assert self.type_id == VpyTypeId.str.value

        data = cast(pointer(self.data), POINTER(VpyValueDataString)).contents
        data = cast(
            pointer(self.data), POINTER(VpyValueDataString.derive_type(data.len))
        ).contents

        return bytearray(data.value).decode()

    def interpreted_expect_function(self) -> str:
        assert self.type_id == VpyTypeId.function.value

        return bytes(self.data).decode()

    def to_python(self) -> object:
        match self.type_id:
            case VpyTypeId.none.value:
                return None
            case VpyTypeId.int.value:
                return self.expect_int()
            case VpyTypeId.bool.value:
                return self.expect_bool()
            case VpyTypeId.float.value:
                return self.expect_float()
            case VpyTypeId.str.value:
                return self.expect_str()
            case x:
                raise RuntimeError(f"unknown compiler value type: {x}")

    def interpreted_to_python(self, x: "Interpreter") -> object:
        match self.type_id:
            case VpyTypeId.function.value:
                return x.functions[self.interpreted_expect_function()]
            case _:
                return self.to_python()
