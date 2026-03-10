from .NSK import NSK_File, parse_nsk, parse_nsk_bytes


def import_nsk(*args, **kwargs):
    # Lazy import to avoid addon startup importing exporter dependencies.
    from .importer import import_nsk as _import_nsk

    return _import_nsk(*args, **kwargs)


def export_nsk(*args, **kwargs):
    # Lazy import to avoid addon startup importing exporter dependencies.
    from .exporter import export_nsk as _export_nsk

    return _export_nsk(*args, **kwargs)


__all__ = [
    "NSK_File",
    "parse_nsk",
    "parse_nsk_bytes",
    "import_nsk",
    "export_nsk",
]
