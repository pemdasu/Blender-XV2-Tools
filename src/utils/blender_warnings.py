import contextlib
from collections.abc import Iterator

import bpy

_shown_warning_keys: set[str] = set()


def show_warning(message: str, *, key: str | None = None) -> None:
    warning_key = key or message
    if warning_key in _shown_warning_keys:
        return
    _shown_warning_keys.add(warning_key)

    print(f"[XV2 Warning] {message}")

    try:
        window_manager = bpy.context.window_manager
    except (AttributeError, RuntimeError):
        return

    if window_manager is None:
        return

    def draw(self, _context):
        self.layout.label(text=message)

    try:
        window_manager.popup_menu(draw, title="XV2 Warning", icon="ERROR")
    except (AttributeError, RuntimeError, TypeError):
        return


@contextlib.contextmanager
def warn_on_error(
    message: str,
    *exceptions: type[BaseException],
    key: str | None = None,
) -> Iterator[None]:
    try:
        yield
    except exceptions as error:
        error_text = str(error)
        if error_text:
            show_warning(f"{message}: {error_text}", key=key or message)
        else:
            show_warning(message, key=key or message)
