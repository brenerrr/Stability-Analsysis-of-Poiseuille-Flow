from dash import Dash, html, dcc, Output, Input, State, Patch, no_update, ctx
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from app_layout import *
import methods as m

# **************************** Default parameters ****************************
h = 2
U_func = lambda x: 1 - x**2
Uyy_func = lambda x: -2
Uy_func = lambda x: -2 * x

coeffs_y = (
    (-1 / 2, 0, 1 / 2),
    (1 / 12, -2 / 3, 0, 2 / 3, -1 / 12),
)

coeffs_yy = (
    (1, -2, 1),
    (-1 / 12, 16 / 12, -30 / 12, 16 / 12, -1 / 12),
    (2 / 180, -27 / 180, 270 / 180, -490 / 180, 270 / 180, -27 / 180, 2 / 180),
)

coeffs_yyyy = (
    (1, -4, 6, -4, 1),
    (-1 / 6, 12 / 6, -39 / 6, 56 / 6, -39 / 6, 12 / 6, -1 / 6),
)


# *****************************************************************************


app = Dash(
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        dbc.icons.BOOTSTRAP,
        dbc.icons.FONT_AWESOME,
    ]
)

app.layout = [
    html.Div(
        id="main-container",
        className="container d-flex flex-column pt-5 pb-3",
        children=[
            title,
            controls,
            plots,
            config_contour,
            config_resolvent,
            config_growth,
            dcc.Store(id="data-alpha"),
            dcc.Store(id="data-beta"),
            dcc.Store(id="data-re"),
            dcc.Store(id="data-n"),
            dcc.Store(id="data-e_val"),
            dcc.Store(id="data-e_vec"),
            dcc.Store(id="data-growth"),
            dcc.Store(id="data-resolvent"),
            dcc.Store(id="data-resolvent_contour"),
        ],
    )
]


# Update alpha
@app.callback(
    Output("data-alpha", "data"),
    Input("input-alpha", "value"),
    prevent_initial_call=False,
)
def update_alpha(alpha):
    return float(alpha)


# Update beta
@app.callback(
    Output("data-beta", "data"),
    Input("input-beta", "value"),
    prevent_initial_call=False,
)
def update_beta(beta):
    return float(beta)


# Update re
@app.callback(
    Output("data-re", "data"), Input("input-re", "value"), prevent_initial_call=False
)
def update_beta(re):
    return float(re)


# Update n
@app.callback(
    Output("data-n", "data"), Input("input-n", "value"), prevent_initial_call=False
)
def update_beta(n):
    return float(n)


# Solve Orr Sommerfeld
@app.callback(
    Output("data-e_val", "data"),
    Output("data-e_vec", "data"),
    State("data-alpha", "data"),
    State("data-beta", "data"),
    State("data-re", "data"),
    State("data-n", "data"),
    Input("button-spectrum", "n_clicks"),
    running=[(Output("button-spectrum", "disabled"), True, False)],
    prevent_initial_call=True,
)
def solve_orr_sommerfeld(alpha, beta, re, n, clicks):
    print("Solving Orr Somerfeld")

    y = np.linspace(-h / 2, h / 2, n, endpoint=True)
    dy = y[1] - y[0]

    scaled_coeffs_y = [np.array(coeffs) / dy for coeffs in coeffs_y]
    scaled_coeffs_yy = [np.array(coeffs) / dy**2 for coeffs in coeffs_yy]
    scaled_coeffs_yyyy = [np.array(coeffs) / dy**4 for coeffs in coeffs_yyyy]

    U = U_func(y)
    Uy = Uy_func(y)
    Uyy = Uyy_func(y)

    Dyy = m.build_diff_operator(scaled_coeffs_yy, n)
    Dyyyy = m.build_diff_operator(scaled_coeffs_yyyy, n)
    Dy = m.build_diff_operator(scaled_coeffs_y, n)

    # Use biased scheme at boundaries
    Dy[0, [0, 1, 2]] = -3 / 2, 2, -1 / 2
    Dy[n - 1, [n - 1, n - 2, n - 3]] = 3 / 2, -2, 1 / 2
    Dy[[0, n - 1], :] /= dy

    omega, e_val, e_vec, L1 = m.solve_orr_sommerfeld(
        alpha,
        beta,
        re,
        U,
        Uy,
        Uyy,
        Dyy,
        Dyyyy,
        dy,
    )
    # Clean inf and nan values
    mask = np.isnan(omega) | np.isinf(omega)

    omega_clean = omega[~mask]
    e_vec_clean = e_vec[:, ~mask]
    e_val_clean = e_val[~mask]

    sort_i = omega_clean.imag.argsort()[::-1]
    omega_sorted = omega_clean[sort_i]
    e_vec_sorted = e_vec_clean[:, sort_i]
    e_val_sorted = e_val_clean[sort_i]
    print("Done solving Orr Sommerfeld")

    return (
        e_val_sorted.real,
        e_val_sorted.imag,
    ), (
        e_vec_sorted.real,
        e_vec_sorted.imag,
    )


# Update spectrum plot
@app.callback(
    Output("fig-spectrum", "figure"),
    Input("data-e_val", "data"),
    Input("data-resolvent_contour", "data"),
    prevent_initial_call=True,
)
def update_spectrum_plot(e_val, resolvent_contour):
    patch = Patch()
    if ctx.triggered_id == "data-resolvent_contour":
        print("Updating spectrum resolvent contours")
        R, (fr, fi) = resolvent_contour
        patch["data"][1] = go.Contour(
            x=fr,
            y=fi,
            z=np.log(R),
            colorscale="Greys",
            ncontours=30,
            contours=dict(coloring="lines"),
            line_width=1,
            showscale=False,
            hoverinfo="skip",
        )
        print("Updating spectrum resolvent contours finished")
        return patch

    elif ctx.triggered_id == "data-e_val":
        print("Updating spectrum plot")
        e_val = np.array(e_val)

        omega = e_val[0] + e_val[1] * 1j
        omega = omega * 1j
        scatter = go.Scatter(x=omega.real, y=omega.imag, mode="markers")
        patch["data"][0] = scatter
        print("Updating spectrum plot finished")
        return patch

    elif ctx.triggered_id == "fig-spectrum":
        return patch


@app.callback(
    Output("fig-vec", "figure"),
    Input("fig-spectrum", "hoverData"),
    State("data-e_vec", "data"),
    State("data-n", "data"),
    prevent_initial_call=True,
)
def update_vec_plot(hover, e_vec, n):
    if hover is None:
        return patch

    patch = Patch()
    print("Updating spectrum hover")

    i = hover["points"][0]["pointNumber"]

    e_vec = np.array(e_vec)
    e_vec = e_vec[0, :, i] + e_vec[1, :, i] * 1j
    n = int(n)

    v = e_vec[:n]
    vort = e_vec[n:]
    v_real = go.Scatter(
        y=v.real, line_color="#58D68D", showlegend=False, mode="lines", name="real"
    )
    v_imag = go.Scatter(
        y=v.imag,
        line_color="#58D68D",
        line_dash="dash",
        mode="lines",
        showlegend=False,
        name="imag",
    )
    v_abs = go.Scatter(
        y=np.abs(v),
        line_color="rgba(0,0,0,0.4)",
        name="abs",
        mode="lines",
        showlegend=False,
    )

    vort_real = go.Scatter(
        y=vort.real,
        line_color="#58D68D",
        mode="lines",
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
        name="real",
    )
    vort_imag = go.Scatter(
        y=vort.imag,
        line_color="#58D68D",
        line_dash="dash",
        mode="lines",
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
        name="imag",
    )
    vort_abs = go.Scatter(
        y=np.abs(vort),
        line_color="rgba(0,0,0,0.4)",
        name="abs",
        mode="lines",
        showlegend=False,
        xaxis="x2",
        yaxis="y2",
    )
    patch["data"][0] = v_real
    patch["data"][1] = v_imag
    patch["data"][2] = v_abs
    patch["data"][3] = vort_real
    patch["data"][4] = vort_imag
    patch["data"][5] = vort_abs
    print("Finishing updating spectrum hover")
    return patch


@app.callback(
    Output("data-growth", "data"),
    Input("button-growth", "n_clicks"),
    State("data-e_val", "data"),
    State("data-e_vec", "data"),
    State("data-alpha", "data"),
    State("data-beta", "data"),
    State("data-n", "data"),
    State("input-00_growth", "value"),
    State("input-01_growth", "value"),
    State("input-02_growth", "value"),
    running=[(Output("button-growth", "disabled"), True, False)],
    prevent_initial_call=True,
)
def calculate_transient_growth(
    n_clicks, e_val, e_vec, alpha, beta, n, t0, tN, tNPoints
):
    if len(e_val) == 0:
        return no_update
    print("Calculating transient growth")
    e_val = np.array(e_val)
    e_val = e_val[0] + e_val[1] * 1j
    e_vec = np.array(e_vec)
    e_vec = e_vec[0] + e_vec[1] * 1j
    alpha = float(alpha)
    beta = float(beta)
    n = int(n)

    y = np.linspace(-h / 2, h / 2, n, endpoint=True)
    dy = y[1] - y[0]
    scaled_coeffs_yy = [np.array(coeffs) / dy**2 for coeffs in coeffs_yy]
    t = np.linspace(t0, tN, tNPoints)
    G = m.calculate_transient_growth(alpha, beta, n, e_vec, e_val, scaled_coeffs_yy, t)
    print("Finished calculating transient growth")

    return (G, t)


@app.callback(
    Output("fig-growth", "figure"),
    Input("data-growth", "data"),
    prevent_initial_call=True,
)
def update_growth_plot(data):
    print("Updating transient growth plot")
    G, t = data
    patch = Patch()
    patch["data"][0] = go.Scatter(x=t, y=G, mode="lines")
    print("Finishing updating transient growth plot")
    return patch


@app.callback(
    Output("data-resolvent", "data"),
    Input("button-resolvent", "n_clicks"),
    State("data-e_val", "data"),
    State("data-e_vec", "data"),
    State("data-alpha", "data"),
    State("data-beta", "data"),
    State("data-n", "data"),
    State("input-00_resolvent", "value"),
    State("input-01_resolvent", "value"),
    State("input-02_resolvent", "value"),
    running=[(Output("button-resolvent", "disabled"), True, False)],
    prevent_initial_call=True,
)
def calculate_resolvent_norm(
    n_clicks, e_val, e_vec, alpha, beta, n, or0, orN, orNpoints
):
    if len(e_val) == 0:
        return no_update
    freqs = np.linspace(or0, orN, orNpoints)
    R = calculate_resolvent(n_clicks, e_val, e_vec, alpha, beta, n, freqs, [0])[0]
    return (R, freqs)


@app.callback(
    Output("data-resolvent_contour", "data"),
    Input("button-resolvent_contour", "n_clicks"),
    State("data-e_val", "data"),
    State("data-e_vec", "data"),
    State("data-alpha", "data"),
    State("data-beta", "data"),
    State("data-n", "data"),
    State("input-00_contour", "value"),
    State("input-01_contour", "value"),
    State("input-02_contour", "value"),
    State("input-10_contour", "value"),
    State("input-11_contour", "value"),
    State("input-12_contour", "value"),
    running=[(Output("button-resolvent_contour", "disabled"), True, False)],
    prevent_initial_call=True,
)
def calculate_resolvent_contour(
    n_clicks, e_val, e_vec, alpha, beta, n, or0, orN, orNpoints, oi0, oiN, oiNpoints
):
    if len(e_val) == 0:
        return no_update
    print("Calculating resolvent contours")
    freqs_r = np.linspace(or0, orN, orNpoints)
    freqs_i = np.linspace(oi0, oiN, oiNpoints)
    R = calculate_resolvent(n_clicks, e_val, e_vec, alpha, beta, n, freqs_r, freqs_i)
    print("Finished calculating resolvent contours")
    return (R, (freqs_r, freqs_i))


def calculate_resolvent(n_clicks, e_val, e_vec, alpha, beta, n, freqs_real, freqs_imag):
    print("Calculating resolvent")
    e_val = np.array(e_val)
    e_val = e_val[0] + e_val[1] * 1j
    e_vec = np.array(e_vec)
    e_vec = e_vec[0] + e_vec[1] * 1j
    alpha = float(alpha)
    beta = float(beta)
    n = int(n)

    from methods import calculate_resolvent

    y = np.linspace(-h / 2, h / 2, n, endpoint=True)
    dy = y[1] - y[0]
    scaled_coeffs_yy = [np.array(coeffs) / dy**2 for coeffs in coeffs_yy]
    R = m.calculate_resolvent(
        alpha,
        beta,
        n,
        e_vec,
        e_val,
        scaled_coeffs_yy,
        freqs_real,
        freqs_imag,
    )
    print("Finished calculating resolvent")
    return R


@app.callback(
    Output("fig-resolvent", "figure"),
    Input("data-resolvent", "data"),
    prevent_initial_call=True,
)
def update_resolvent_norm_plot(data):
    print("Updating resolvent plot")
    R, freqs = data
    patch = Patch()
    patch["data"][0] = go.Scatter(x=freqs, y=R, mode="lines")
    print("Finishing updating resolvent plot")
    return patch


@app.callback(
    Output("modal-contour", "is_open"),
    Output("modal-growth", "is_open"),
    Output("modal-resolvent", "is_open"),
    Input("button-config_resolvent_contour", "n_clicks"),
    Input("button-config_growth", "n_clicks"),
    Input("button-config_resolvent", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_modal(n_clicks, n_clicks1, nclicks_2):
    out = {
        "button-config_resolvent_contour": [True, False, False],
        "button-config_growth": [False, True, False],
        "button-config_resolvent": [False, False, True],
    }
    return out[ctx.triggered_id]


@app.callback(
    Output("button-spectrum", "children", allow_duplicate=True),
    Output("button-resolvent_contour", "children", allow_duplicate=True),
    Output("button-growth", "children", allow_duplicate=True),
    Output("button-resolvent", "children", allow_duplicate=True),
    Input("button-spectrum", "disabled"),
    State("button-spectrum", "children"),
    Input("button-resolvent_contour", "disabled"),
    State("button-resolvent_contour", "children"),
    Input("button-growth", "disabled"),
    State("button-growth", "children"),
    Input("button-resolvent", "disabled"),
    State("button-resolvent", "children"),
    prevent_initial_call=True,
)
def update_button(
    spec_disabled,
    spec,
    contour_disabled,
    contour,
    growth_disabled,
    growth,
    resolvent_disabled,
    resolvent,
):
    selector = {
        "button-spectrum": (spec, spec_disabled, 0),
        "button-resolvent_contour": (contour, contour_disabled, 1),
        "button-growth": (growth, growth_disabled, 2),
        "button-resolvent": (resolvent, resolvent_disabled, 3),
    }
    button, is_disabled, i = selector[ctx.triggered_id]

    out = [no_update for _ in range(4)]

    if is_disabled:
        out[i] = [dbc.Spinner(size="sm"), " "] + button
    else:
        out[i] = button[1:]

    return out


if __name__ == "__main__":
    app.run(port=8050, debug=True)
