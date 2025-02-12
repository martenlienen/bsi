import functools
from io import BytesIO

import matplotlib
import matplotlib.pyplot as pp
import PIL


def render_figure(fig: pp.Figure) -> PIL.Image:
    """Render a matplotlib figure into a Pillow image."""
    buf = BytesIO()
    fig.savefig(buf, **{"format": "rgba"})
    return PIL.Image.frombuffer(
        "RGBA", fig.canvas.get_width_height(), buf.getbuffer(), "raw", "RGBA", 0, 1
    )


def render_and_close(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Use non-interative backend to avoid errors in multi-process rendering. We
        # initialize it here, to ensure that it is set in every subprocess.
        matplotlib.use("Agg")

        fig = f(*args, **kwargs)
        if fig is None:
            return None
        img = render_figure(fig)
        pp.close(fig)
        return img

    return wrapper
