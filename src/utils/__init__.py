import struct

import bpy


def read_cstring(data: bytes, offset: int) -> str:
    out = bytearray()
    while offset < len(data) and data[offset] != 0:
        out.append(data[offset])
        offset += 1
    return out.decode("utf8", errors="ignore")


def half_to_float(half_bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", half_bits))[0]


def float_to_half(value: float) -> int:
    return struct.unpack("<H", struct.pack("<e", float(value)))[0]


def remove_unused_vertex_groups(obj: bpy.types.Object):
    if obj.type != "MESH" or not obj.vertex_groups:
        return

    used_indices: set[int] = set()
    for vertex in obj.data.vertices:
        for group_element in vertex.groups:
            used_indices.add(group_element.group)

    # Remove in reverse index order so remaining indices stay valid.
    for index in range(len(obj.vertex_groups) - 1, -1, -1):
        if index not in used_indices:
            obj.vertex_groups.remove(obj.vertex_groups[index])
