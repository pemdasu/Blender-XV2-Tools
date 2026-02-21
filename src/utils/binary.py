from __future__ import annotations

import struct


def is_valid_offset(data: bytes, offset: int, size: int = 1) -> bool:
    return 0 <= offset <= len(data) - size


def u16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<H", data, offset)[0]


def i16(data: bytes, offset: int) -> int:
    return struct.unpack_from("<h", data, offset)[0]


def u32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def u64(data: bytes, offset: int) -> int:
    return struct.unpack_from("<Q", data, offset)[0]


def i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def f32(data: bytes, offset: int) -> float:
    return struct.unpack_from("<f", data, offset)[0]


__all__ = [
    "f32",
    "i16",
    "i32",
    "is_valid_offset",
    "u16",
    "u32",
    "u64",
]
