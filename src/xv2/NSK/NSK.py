from dataclasses import dataclass

from ...utils.binary import u32
from ..EMD import EMD_File, parse_emd_bytes
from ..ESK import ESK_File, parse_esk_bytes

ESK_MAGIC = b"#ESK"
EMD_MAGIC = b"#EMD"
NSK_EMD_OFFSET_ADDRESS = 20


@dataclass
class NSK_File:
    esk_file: ESK_File
    emd_file: EMD_File
    emd_offset: int


def parse_nsk_bytes(data: bytes) -> NSK_File:
    if len(data) < 24:
        raise ValueError("NSK file is too small.")

    esk_offset = data.find(ESK_MAGIC)
    if esk_offset == -1:
        raise ValueError('Could not locate "#ESK" signature in NSK file.')
    if esk_offset != 0:
        raise ValueError("NSK #ESK signature is not at offset 0.")

    emd_offset = u32(data, NSK_EMD_OFFSET_ADDRESS)
    if emd_offset <= 0 or emd_offset + 4 > len(data):
        raise ValueError("Invalid EMD offset in NSK header.")
    if data[emd_offset : emd_offset + 4] != EMD_MAGIC:
        raise ValueError('Could not locate "#EMD" signature at header-defined NSK offset.')

    esk_file = parse_esk_bytes(data)
    emd_file = parse_emd_bytes(data[emd_offset:])
    return NSK_File(esk_file=esk_file, emd_file=emd_file, emd_offset=emd_offset)


def parse_nsk(path: str) -> NSK_File:
    with open(path, "rb") as file_handle:
        data = file_handle.read()
    return parse_nsk_bytes(data)
