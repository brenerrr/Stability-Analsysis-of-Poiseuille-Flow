import itertools
from dash import html, dcc
import os
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import json
import plotly.io as pio
from plotly.subplots import make_subplots

template = go.layout.Template()

with open("plotly_template.json") as f:
    template_dict = json.load(f)

template.layout.update(**template_dict["layout"])
for k, v in template_dict["data"].items():
    template.data[k] = [v]

pio.templates.default = template

fig_spectrum = go.Figure()
fig_spectrum.update_layout(
    xaxis_title="omega real",
    yaxis_title="omega imag",
    xaxis_range=[0, 1],
    yaxis_range=[-1, 0.1],
)
fig_vec = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    shared_xaxes=True,
    x_title="y",
    column_titles=["v", "vort"],
)
fig_vec.add_traces([go.Scatter() for _ in range(3)], 1, 1)
fig_vec.add_traces([go.Scatter() for _ in range(3)], 1, 2)

fig_growth = go.Figure()
fig_growth.update_layout(yaxis_title="Energy Amplification", xaxis_title="time")

fig_resolvent = go.Figure()
fig_resolvent.update_layout(
    yaxis_type="log", yaxis_title="Resolvent Operator Norm", xaxis_title="omega real"
)


title = html.H1(children="Stability Analysis of Plane Poiseuille Flow")

controls = html.Div(
    className="card d-flex flex-column",
    children=[
        html.Div(
            className="card-body d-flex flex-column ",
            children=[
                html.Div(
                    className="container card-body",
                    children=[
                        html.Div(
                            className="row",
                            children=[
                                html.Div(
                                    "Streamwise Wavenumber",
                                    className="col-md-2 me-3",
                                ),
                                html.Div(
                                    "Spanwise Wavenumber",
                                    className="col-md-2 me-3",
                                ),
                                html.Div(
                                    "Reynods Number",
                                    className="col-md-2 me-3",
                                ),
                                html.Div(
                                    "N mesh points",
                                    className="col-md-2 me-3",
                                ),
                            ],
                        ),
                        html.Div(
                            className="row",
                            children=[
                                dcc.Input(
                                    id="input-alpha",
                                    placeholder=r"α - streamwise wavenumber",
                                    value=1.0,
                                    className="col-md-2 me-3",
                                ),
                                dcc.Input(
                                    id="input-beta",
                                    placeholder=r"β - spanwise wavenumber",
                                    value=0.25,
                                    className="col-md-2 me-3",
                                ),
                                dcc.Input(
                                    id="input-re",
                                    placeholder=r"Re - Reynolds number",
                                    value=2000,
                                    className="col-md-2 me-3",
                                ),
                                dcc.Input(
                                    id="input-n",
                                    placeholder=r"N mesh points",
                                    value=201,
                                    className="col-md-2 me-3",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        )
    ],
)

plots = html.Div(
    className="d-flex flex-row mt-3 mb-3 gap-3",
    id="plots-container",
    children=[
        html.Div(
            className="container card d-flex flex-column",
            children=[
                html.Div(
                    className="container card-body d-flex flex-row",
                    children=[
                        dbc.Button(
                            "Spectrum",
                            id="button-spectrum",
                            className="align-self-start me-2",
                        ),
                        dbc.Button(
                            "Resolvent Contours",
                            id="button-resolvent_contour",
                            className="align-self-start me-2",
                        ),
                        dbc.Button(
                            html.I(className="bi bi-gear align-self-center me-2 "),
                            className="btn btn-config",
                            id="button-config_resolvent_contour",
                        ),
                    ],
                ),
                dcc.Graph(
                    id="fig-spectrum",
                    config={"scrollZoom": True},
                    figure=fig_spectrum,
                    responsive=True,
                    style={"width": "99%", "height": "99%"},
                ),
                dcc.Graph(
                    id="fig-vec",
                    config={"scrollZoom": True},
                    figure=fig_vec,
                    responsive=True,
                    style={"width": "99%", "height": "99%"},
                ),
            ],
        ),
        html.Div(
            className="container d-flex flex-column p-0",
            children=[
                html.Div(
                    className="container card d-flex flex-column flex-fill",
                    children=[
                        html.Div(
                            className="container card-body d-flex flex-column flex-fill",
                            children=[
                                html.Div(
                                    className="container d-flex flex-row",
                                    children=[
                                        dbc.Button(
                                            "Transient",
                                            id="button-growth",
                                            className="align-self-start me-2",
                                        ),
                                        dbc.Button(
                                            html.I(
                                                className="bi bi-gear align-self-center me-2 "
                                            ),
                                            className="btn btn-config",
                                            id="button-config_growth",
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="fig-growth",
                                    config={"scrollZoom": True},
                                    figure=fig_growth,
                                    responsive=True,
                                    style={"width": "99%", "height": "99%"},
                                ),
                            ],
                        ),
                    ],
                ),
                html.Div(
                    className="container card d-flex flex-column mt-3 align-self-star flex-fill",
                    children=[
                        html.Div(
                            className="container card-body d-flex flex-column flex-fill",
                            children=[
                                html.Div(
                                    className="container d-flex flex-row",
                                    children=[
                                        dbc.Button(
                                            "Resolvent Norm",
                                            id="button-resolvent",
                                            className="align-self-start me-2",
                                        ),
                                        dbc.Button(
                                            html.I(
                                                className="bi bi-gear align-self-center me-2 "
                                            ),
                                            className="btn btn-config",
                                            id="button-config_resolvent",
                                        ),
                                    ],
                                ),
                                dcc.Graph(
                                    id="fig-resolvent",
                                    config={"scrollZoom": True},
                                    figure=fig_resolvent,
                                    responsive=True,
                                    style={"width": "99%", "height": "99%"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def create_popup(title, id, headers, default_values):
    headers_html = []
    values_html = []
    for i, (header, values) in enumerate(zip(headers, default_values)):
        header_html = [html.Div(s, className="col-sm text-start") for s in header]
        v_html = [
            dbc.Input(value=v, type="number", className="col", id=f"input-{i}{j}_{id}")
            for j, v in enumerate(values)
        ]

        headers_html.append(html.Div(className="row", children=header_html))

        values_html.append(html.Div(className="row flex-fill mb-3", children=v_html))

    final_html = list(itertools.chain(*zip(headers_html, values_html)))

    popup = dbc.Modal(
        className="container",
        centered=True,
        is_open=False,
        children=[
            dbc.ModalHeader(dbc.ModalTitle(title)),
            dbc.ModalBody(children=final_html),
        ],
        id=f"modal-{id}",
    )
    return popup


config_contour = create_popup(
    "Resolvent Contours Config",
    "contour",
    [["omega_r 0", "omega_r N", "N points"], ["omega_i 0", "omega_i N", "N points"]],
    [[-0.1, 1.1, 15], [-1.1, 0.1, 15]],
)

config_growth = create_popup(
    "Transient Growth Config",
    "growth",
    [["t 0", "t N", "N points"]],
    [[0, 50, 50]],
)

config_resolvent = create_popup(
    "Resolvent Norm Config",
    "resolvent",
    [["omega_r 0", "omega_r N", "N points"]],
    [[-0.5, 1.5, 30]],
)
