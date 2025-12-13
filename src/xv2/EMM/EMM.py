from __future__ import annotations

import os
import struct
from dataclasses import dataclass

EMM_SIGNATURE = 1296909603


@dataclass
class EMMParameter:
    name: str
    type: int
    value: str


@dataclass
class EMMMaterial:
    name: str
    shader: str
    params: list[EMMParameter]


def _read_cstr(data: bytes, offset: int, length: int) -> str:
    raw = data[offset : offset + length]
    return raw.split(b"\x00", 1)[0].decode("ascii", "ignore")


def _parse_parameters(data: bytes, offset: int, count: int) -> list[EMMParameter]:
    params: list[EMMParameter] = []
    for _ in range(count):
        name = _read_cstr(data, offset + 0, 32)
        ptype = struct.unpack_from("<i", data, offset + 32)[0]
        match ptype:
            case 0:  # float
                value = str(struct.unpack_from("<f", data, offset + 36)[0])
            case 65537:  # int
                value = str(struct.unpack_from("<i", data, offset + 36)[0])
            case 131074:  # float2 (stored as float)
                value = str(struct.unpack_from("<f", data, offset + 36)[0])
            case 196611:  # bool/int
                ival = struct.unpack_from("<i", data, offset + 36)[0]
                value = "true" if ival == 1 else "false" if ival == 0 else str(ival)
            case _:  # fallback
                value = str(struct.unpack_from("<i", data, offset + 36)[0])
        params.append(EMMParameter(name=name, type=ptype, value=value))
        offset += 40
    return params


def parse_emm(path: str) -> list[EMMMaterial]:
    with open(path, "rb") as f:
        data = f.read()

    if len(data) < 16 or struct.unpack_from("<I", data, 0)[0] != EMM_SIGNATURE:
        raise ValueError("Invalid EMM signature")

    header_size = struct.unpack_from("<h", data, 12)[0]
    table_offset = struct.unpack_from("<I", data, 12)[0]
    count = struct.unpack_from("<I", data, table_offset)[0]

    materials: list[EMMMaterial] = []
    offset = table_offset + 4
    for _ in range(count):
        entry_rel = struct.unpack_from("<I", data, offset)[0]
        offset += 4
        if entry_rel == 0:
            continue
        mat_off = entry_rel + header_size
        name = _read_cstr(data, mat_off + 0, 32)
        shader = _read_cstr(data, mat_off + 32, 32)
        param_count = struct.unpack_from("<h", data, mat_off + 64)[0]
        params = _parse_parameters(data, mat_off + 68, param_count)
        materials.append(EMMMaterial(name=name, shader=shader, params=params))

    return materials


def locate_emm(path: str) -> str | None:
    base_dir = os.path.dirname(path)
    stem, _ext = os.path.splitext(os.path.basename(path))
    for cand in (os.path.join(base_dir, f"{stem}.emm"),):
        if os.path.isfile(cand):
            return cand
    return None


__all__ = ["parse_emm", "locate_emm", "EMMMaterial", "EMMParameter"]
