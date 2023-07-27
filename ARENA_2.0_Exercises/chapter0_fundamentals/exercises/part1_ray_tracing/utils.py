import plotly.express as px
import plotly.graph_objects as go
import torch as t

def render_lines_with_plotly(lines: t.Tensor, bold_lines: t.Tensor = t.Tensor()):
    '''
    Plot any number of line segments in 3D.

    lines: shape (num_lines, num_points=2, num_dims=3).

    bold_lines: same shape as lines. If supplied, these lines will be rendered in black on top of the other lines.
    '''
    fig = go.Figure(layout=dict(showlegend=False, title="3D rays", height=600, width=600))
    for line in lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines")
    for line in bold_lines:
        X, Y, Z = line.T
        fig.add_scatter3d(x=X, y=Y, z=Z, mode="lines", line_width=5, line_color="black")
    fig.show()


def setup_widget_fig_ray():
    fig = go.FigureWidget(go.Scatter(x=[], y=[]), layout={"width": 600, "height": 500})
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12)
    fig.add_scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
    fig.update_layout(showlegend=False, xaxis_range=[-1.5, 2.5], yaxis_range=[-1.5, 2.5], template="simple_white")
    return fig


def setup_widget_fig_triangle(x, y, z):
    return go.FigureWidget(
        data=[
            go.Scatter(x=x, y=y, mode="markers+text", text=["A", "B", "C"], textposition="middle left", textfont_size=18, marker_size=12),
            go.Scatter(x=[*x, x[0]], y=[*y, y[0]], mode="lines"),
            go.Scatter(x=[], y=[], mode="markers", marker_size=12, marker_symbol="x")
        ],
        layout=dict(
            title="Barycentric coordinates illustration", showlegend=False,
            xaxis_range=[-3, 8], yaxis_range=[-2, 5.5], height=600, width=800, template="simple_white"
        )
    )