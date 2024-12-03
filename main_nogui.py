# %%
import os

# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from IPython.display import display, HTML
import plotly
from methods import (
    solve_orr_sommerfeld,
    build_diff_operator,
    calculate_resolvent,
    calculate_transient_growth,
)

plotly.offline.init_notebook_mode()
display(
    HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    )
)
pio.templates.default = "plotly_white"

# ********************************** Inputs **********************************

n = 301
h = 2
alpha = 1.0
beta = 0.25
Re = 2000

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
# ***************************************************************************

y = np.linspace(-h / 2, h / 2, n, endpoint=True)
U = 1 - y**2
Uyy = -2
Uy = -2 * y

dy = y[1] - y[0]
coeffs_y = [np.array(coeffs) / dy for coeffs in coeffs_y]
coeffs_yy = [np.array(coeffs) / dy**2 for coeffs in coeffs_yy]
coeffs_yyyy = [np.array(coeffs) / dy**4 for coeffs in coeffs_yyyy]


Dyy = build_diff_operator(coeffs_yy, n)
Dyyyy = build_diff_operator(coeffs_yyyy, n)
Dy = build_diff_operator(coeffs_y, n)

# Use biased scheme at boundaries
Dy[0, [0, 1, 2]] = -3 / 2, 2, -1 / 2
Dy[n - 1, [n - 1, n - 2, n - 3]] = 3 / 2, -2, 1 / 2
Dy[[0, n - 1], :] /= dy


omega, e_val, e_vec, L1 = solve_orr_sommerfeld(
    alpha,
    beta,
    Re,
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

# Calculate G(t) || FV exp(lambda t) (FV)^-1 ||^2
# Dirty trick so that it is possible to perform a cholesky decomposition on Q
t = np.linspace(0, 50, 50)
max_energy = calculate_transient_growth(
    alpha, beta, n, e_vec_sorted, e_val_sorted, coeffs_yy, t
)

# Resolvent norm
freqs = np.linspace(-0.5, 1.5, 50)
resolvent_norm = calculate_resolvent(
    alpha, beta, n, e_vec_sorted, e_val_sorted, coeffs_yy, freqs, [0]
)

# Data for contours in spectrum plot
freqs_i = np.linspace(-1, 0.1, 30)
freqs_r = np.linspace(0, 1, 30)

resolvent_contours = calculate_resolvent(
    alpha, beta, n, e_vec_sorted, e_val_sorted, coeffs_yy, freqs_r, freqs_i
)

# *********************************** Plots ***********************************

# Resolvent
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=freqs,
        y=resolvent_norm[0],
    )
)
fig.update_layout(
    yaxis=dict(
        title="Resolvent Norm",
        type="log",
        range=[0, 3],
    ),
    xaxis_title=r"$\omega_r$",
)
fig.show()
fig.write_image("resolvent_norm.png", scale=2)

# G plot
fig = go.Figure(
    go.Scatter(
        x=t,
        y=max_energy,
    ),
    layout=go.Layout(
        yaxis_title="G",
        xaxis_title="t",
        yaxis_type="log",
    ),
)
fig.update_layout(
    font=dict(family="CMU Serif", size=18, color="black"),
    width=500,
    height=500,
    margin=dict(l=0, t=0, b=0, r=0),
)
fig.write_image("transient_growth.png", scale=2)
fig.show()


# Eigenvalues plot
n_eig_vec_show = min(10, len(omega_sorted))
n_omega_show = min(100, len(omega_sorted))

v = e_vec_sorted[:n, :n_eig_vec_show]
vort = e_vec_sorted[n:, :n_eig_vec_show]

omega_plot = omega_sorted[:n_omega_show]

fig = go.Figure()

fig_eigs = fig.add_scatter(
    x=omega_plot.real,
    y=omega_plot.imag,
    marker_color="red",
    marker_size=15,
    mode="markers",
    name="Brener",
)

fig_eigs.update_layout(
    yaxis_range=[-1, 0.1],
    xaxis_range=[0, 1],
    yaxis_title="$\omega_i$",
    xaxis_title="$\omega_r$",
    showlegend=True,
    font=dict(family="CMU Serif", size=18, color="black"),
    width=500,
    height=500,
    margin=dict(l=0, t=0, b=0, r=0),
    legend=dict(yanchor="bottom", xanchor="left", x=0, y=0),
)

schmidt_real = np.loadtxt("real_part.dat")
schmidt_imag = np.loadtxt("imag_part.dat")
eigs_schmidt = schmidt_real + schmidt_imag * 1j
fig_eigs.add_scatter(
    x=eigs_schmidt.real,
    y=eigs_schmidt.imag,
    mode="markers",
    marker_symbol="x",
    marker_color="black",
    marker_size=8,
    name="Schmid",
)

fig_eigs.add_trace(
    go.Contour(
        x=freqs_r,
        y=freqs_i,
        z=np.log(resolvent_contours),
        colorscale="Greys",
        ncontours=30,
        contours=dict(coloring="lines"),
        line_width=1,
        showscale=False,
    )
)

fig_eigs.show(config=dict(scrollZoom=True))
fig_eigs.write_image(f"eigenvalues.png", scale=2)

# Eigenvectors plot
fig = make_subplots(
    n_eig_vec_show,
    2,
    shared_xaxes=True,
    column_titles=["$v$", "$\eta$"],
)

for j, var in enumerate([v, vort]):
    if var.shape[1] == 0:
        continue
    for i in range(n_eig_vec_show):
        fig.add_trace(
            go.Scatter(y=var[:, i].real, line_color="#58D68D", showlegend=False),
            row=i + 1,
            col=j + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=var[:, i].imag,
                line_color="#58D68D",
                line_dash="dash",
                showlegend=False,
            ),
            row=i + 1,
            col=j + 1,
        )
        fig.add_trace(
            go.Scatter(
                y=np.abs(var[:, i]),
                line_color="rgba(0,0,0,0.4)",
                name="abs",
                showlegend=False,
            ),
            row=i + 1,
            col=j + 1,
        )


fig.update_layout(
    font=dict(family="CMU Serif", size=18, color="black"),
    margin=dict(l=0, r=0, t=50, b=0),
    height=1500,
)
fig.for_each_annotation(lambda a: a.update(y=1.01))
fig.show()
fig.write_image(f"eigenvecs.png", scale=2)
