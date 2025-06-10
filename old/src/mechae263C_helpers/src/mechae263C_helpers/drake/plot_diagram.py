import copy
import math
import warnings
from io import BytesIO
from itertools import zip_longest
from pathlib import Path
from typing import Sequence, NamedTuple, Any

import fitz
import matplotlib.pyplot as plt
import numpy as np
import pydot
from numpy.typing import NDArray
from mpl_toolkits.axes_grid1 import Divider, Size
from pydrake.systems.drawing import plot_graphviz, plot_system_graphviz
from pydrake.systems.framework import Diagram


def plot_diagram(
    diagram: Diagram, fig_width_in: int | float = 20.0, dpi: int = 300
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure()
    ax: plt.Axes = fig.add_subplot(111)
    for spine in ax.spines:
        ax.spines[spine].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    dot: pydot.Dot = pydot.graph_from_dot_data(diagram.GetGraphvizString())[0]
    pdf = dot.create(format="pdf")
    pdf_data = BytesIO()
    pdf_data.write(pdf)
    pdf_data.seek(0)
    doc = fitz.Document(stream=pdf_data)
    png = doc[0].get_pixmap(dpi=dpi)
    png_data = BytesIO()
    png_data.write(png.tobytes())
    png_data.seek(0)
    img = plt.imread(png_data)
    plt.imshow(img)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    width, height = abs(x_max - x_min), abs(y_max - y_min)
    aspect_ratio = height / width
    fig_height_in = aspect_ratio * fig_width_in
    fig.set_size_inches(fig_width_in, fig_height_in)

    return fig, ax
