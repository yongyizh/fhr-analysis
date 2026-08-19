"""Interactive (zoom / pan / drag) HTML plot of two time-series, via plotly.

Kept dependency-light — numpy + plotly only, NO project imports — so both process
scripts can use it, including the FUNet one whose bare `config`/`data`/`model`
imports from lib/funet/src must not be shadowed by yz modules of the same name.
"""
import numpy as np


def minmax_decimate(t, y, n_bins):
    """Downsample to ~2*n_bins points, keeping the min AND max of each time bin so
    peaks/troughs survive (a plain stride would drop beat spikes). For the HTML only,
    so a long high-rate trace stays small yet still shows structure when zoomed."""
    t = np.asarray(t, float); y = np.asarray(y, float)
    if t.size <= 2 * n_bins:
        return t, y
    edges = np.linspace(0, t.size, n_bins + 1).astype(int)
    idx = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        seg = y[a:b]
        idx.extend(sorted((a + int(np.argmin(seg)), a + int(np.argmax(seg)))))
    return (lambda k: (t[k], y[k]))(np.unique(idx))


def _beat_segments(beats, y_plotted):
    """Vertical beat markers as ONE NaN-separated polyline: x = [b, b, nan, ...],
    y = [lo, hi, nan, ...]. A per-beat layout shape (add_vline) would be cleaner but
    puts thousands of shapes in the page and makes it crawl, so the whole beat train
    goes in a single trace instead. The segments span the plotted data's own min-max,
    which leaves the panel's autorange exactly as the data trace set it, and still
    crosses the full height whenever you are zoomed in.
    """
    bt = np.asarray(beats, dtype=float)
    bt = bt[np.isfinite(bt)]
    yv = np.asarray(y_plotted, dtype=float)
    yv = yv[np.isfinite(yv)]
    if bt.size == 0 or yv.size == 0:
        return np.array([]), np.array([])
    lo, hi = float(yv.min()), float(yv.max())
    if lo == hi:                                  # flat trace: give the ticks some height
        lo, hi = lo - 0.5, hi + 0.5
    x = np.repeat(bt, 3)
    x[2::3] = np.nan                              # nan breaks the line between beats
    y = np.tile(np.array([lo, hi, np.nan]), bt.size)
    return x, y


def save_interactive(out_html, title, t_a, y_a, name_a, t_b, y_b, name_b,
                     y_a_title="signal A", y_b_title="signal B",
                     max_points=200000, x_title="Time (s)"):
    """Write a self-contained zoom/pan/drag HTML: series A (left y) + series B (right y).

    Scroll to zoom, drag to pan, plus a range slider along the bottom. plotly.js is
    inlined so the file works offline. Each trace is peak-preserving-decimated to keep
    it under ~max_points.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    ta, ya = minmax_decimate(t_a, y_a, max_points // 2)
    tb, yb = minmax_decimate(t_b, y_b, max_points // 2)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scattergl(x=ta, y=ya, name=name_a, line=dict(color="green", width=1)),
                  secondary_y=False)
    fig.add_trace(go.Scattergl(x=tb, y=yb, name=name_b,
                               line=dict(color="royalblue", width=1), opacity=0.6),
                  secondary_y=True)
    fig.update_layout(title=title, template="plotly_white", hovermode="x unified",
                      dragmode="pan", legend=dict(orientation="h", y=1.02, x=0))
    fig.update_xaxes(title_text=x_title, rangeslider=dict(visible=True))
    fig.update_yaxes(title_text=y_a_title, secondary_y=False)
    fig.update_yaxes(title_text=y_b_title, secondary_y=True)
    # see save_interactive_multi: the range slider would otherwise lock both y axes
    fig.update_yaxes(fixedrange=False)
    fig.write_html(out_html, include_plotlyjs=True,
                   config={"scrollZoom": True, "displaylogo": False})


def save_interactive_multi(out_html, title, series, max_points=200000,
                           x_title="Time (s)", colors=None):
    """Write a self-contained zoom/pan HTML with one x-linked panel per series.

    `series` is a list of (name, t, y) — or (name, t, y, beats) — tuples; each becomes
    its own stacked panel that shares the x-axis, so scrolling to zoom or dragging to
    pan one panel moves them all together (a range slider along the bottom does the
    same). An optional 4th element `beats` is an array of times drawn as red dashed
    vertical lines on that panel, e.g. detected beats. Each trace is
    peak-preserving-decimated to stay under ~max_points, and plotly.js is inlined so
    the file works offline. Handy for many raw channels (e.g. the 5 FUNet fibers).
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    n = len(series)
    palette = colors or ["green", "royalblue", "firebrick", "darkorange",
                         "purple", "teal", "crimson", "seagreen"]
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True,
                        subplot_titles=[s[0] for s in series],
                        vertical_spacing=max(0.012, 0.30 / max(n, 1)))
    for i, s in enumerate(series, start=1):
        name, t, y = s[0], s[1], s[2]
        beats = s[3] if len(s) > 3 else None
        td, yd = minmax_decimate(t, y, max_points // 2)
        fig.add_trace(go.Scattergl(x=td, y=yd, name=name,
                                   line=dict(color=palette[(i - 1) % len(palette)], width=1)),
                      row=i, col=1)
        if beats is not None and len(beats):
            bx, by = _beat_segments(beats, yd)
            if bx.size:
                # hoverinfo="skip": the segment endpoints are the panel's min/max, not
                # data, so they would just be noise in the unified hover box
                fig.add_trace(go.Scattergl(x=bx, y=by, mode="lines",
                                           name=f"{name} beats", showlegend=False,
                                           hoverinfo="skip",
                                           line=dict(color="red", width=1, dash="dash")),
                              row=i, col=1)
    fig.update_layout(title=title, template="plotly_white", dragmode="pan",
                      height=max(600, 170 * n), hovermode="x unified", showlegend=False)
    fig.update_xaxes(title_text=x_title, row=n, col=1, rangeslider=dict(visible=True))
    # plotly.js silently forces fixedrange=True on every y-axis anchored to an x-axis that
    # carries a range slider, which locks y zoom on the BOTTOM panel only. Setting it back
    # explicitly wins over that default and keeps the slider. Do not drop this line.
    fig.update_yaxes(fixedrange=False)
    fig.write_html(out_html, include_plotlyjs=True,
                   config={"scrollZoom": True, "displaylogo": False})


def _robust_scale(y, pct=99.0):
    """Amplitude used to put unrelated signals on a common visual scale. The 99th
    percentile of |y| rather than max(|y|), so one artifact spike does not flatten the
    whole trace to a line."""
    a = np.abs(np.asarray(y, dtype=float))
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 1.0
    s = float(np.percentile(a, pct))
    return s if s > 0 else (float(a.max()) or 1.0)


def save_interactive_overlay(out_html, title, series, max_points=200000,
                             x_title="Time (s)", offset=2.0, normalize=True,
                             beat_span="full", colors=None, height=650,
                             stats_text=None):
    """Write a self-contained zoom/pan HTML with every series on ONE shared panel.

    Unlike `save_interactive_multi` (one panel per series) this stacks the traces in a
    single panel, series k shifted up by ``k * offset``, so two recordings of the same
    event can be compared beat for beat on one time axis. `series` is a list of
    (name, t, y) — or (name, t, y, beats) — tuples; the first is drawn at the bottom.

    `normalize=True` divides each trace by its own robust amplitude first, so signals
    with wildly different units (a fiber at 1e-3 vs a microphone at 1e3) come out the
    same visual size; the y-axis is then labelled per trace instead of by value, since
    the numbers no longer mean anything. Beats are dashed vertical lines in their own
    trace's colour: ``beat_span="full"`` spans the whole panel, which makes it obvious
    whether two beat trains line up, and ``"signal"`` keeps each set inside its own
    trace's band when the full-height lines get too busy.

    `stats_text` is drawn as a boxed annotation in the top-right of the plotting area —
    for the beat-agreement numbers that go with the picture.
    """
    import plotly.graph_objects as go
    palette = colors or ["royalblue", "firebrick", "green", "darkorange",
                         "purple", "teal", "crimson", "seagreen"]
    if beat_span not in ("full", "signal"):
        raise ValueError(f"beat_span must be 'full' or 'signal', got {beat_span!r}")

    prepared, lo_all, hi_all = [], np.inf, -np.inf
    for k, s in enumerate(series):
        name, t, y = s[0], s[1], s[2]
        y = np.asarray(y, dtype=float)
        if normalize:
            y = y / _robust_scale(y)
        y = y + k * offset
        td, yd = minmax_decimate(t, y, max_points // 2)
        finite = yd[np.isfinite(yd)]
        if finite.size:
            lo_all = min(lo_all, float(finite.min()))
            hi_all = max(hi_all, float(finite.max()))
        prepared.append(dict(name=name, td=td, yd=yd, beats=s[3] if len(s) > 3 else None,
                             color=palette[k % len(palette)], base=k * offset))

    fig = go.Figure()
    for p in prepared:
        fig.add_trace(go.Scattergl(x=p["td"], y=p["yd"], name=p["name"], mode="lines",
                                   line=dict(color=p["color"], width=1)))
    for p in prepared:
        if p["beats"] is None or not len(p["beats"]):
            continue
        span = np.array([lo_all, hi_all]) if beat_span == "full" else p["yd"]
        bx, by = _beat_segments(p["beats"], span)
        if bx.size:
            fig.add_trace(go.Scattergl(x=bx, y=by, mode="lines",
                                       name=f"{p['name']} beats", hoverinfo="skip",
                                       line=dict(color=p["color"], width=1, dash="dash")))

    # the horizontal legend sits above the plot area, so the top margin has to hold BOTH
    # it and the title or they overprint each other
    fig.update_layout(title=dict(text=title, y=0.975, yanchor="top", x=0.5, xanchor="center"),
                      template="plotly_white", dragmode="pan",
                      height=height, hovermode="x unified",
                      margin=dict(t=110),
                      legend=dict(orientation="h", yanchor="bottom", y=1.01,
                                  xanchor="left", x=0))
    if stats_text:
        fig.add_annotation(text=stats_text.replace("\n", "<br>"),
                           xref="paper", yref="paper", x=0.995, y=0.99,
                           xanchor="right", yanchor="top", showarrow=False,
                           align="left", font=dict(size=11, family="monospace"),
                           bgcolor="rgba(255,255,255,0.85)",
                           bordercolor="#999", borderwidth=1, borderpad=5)
    fig.update_xaxes(title_text=x_title, rangeslider=dict(visible=True))
    # see save_interactive_multi: the range slider would otherwise lock the y axis
    fig.update_yaxes(fixedrange=False)
    if normalize:
        fig.update_yaxes(tickmode="array",
                         tickvals=[p["base"] for p in prepared],
                         ticktext=[p["name"] for p in prepared],
                         title_text="(normalised, offset)")
    fig.write_html(out_html, include_plotlyjs=True,
                   config={"scrollZoom": True, "displaylogo": False})


def save_interactive_beats_hr(out_html, title,
                              t_env, env, env_name,
                              t_act, act, act_name,
                              beats_env, beats_act,
                              t_bpm_env, bpm_env, bpm_env_name,
                              t_bpm_act, bpm_act, bpm_act_name,
                              max_points=200000, x_title="Time (s)", height=760,
                              bpm_subtitle="BPM (60/IBI)"):
    """Two x-linked zoom/pan panels for the beat-timing check.

    Top: the two beat-carrying signals (each scaled by its own 99th-percentile
    amplitude, so one artifact spike cannot flatten the trace) with every beat a
    vertical line -- solid red for the env/target train, dashed blue for the
    act/model train. Bottom: both trains' 60/IBI BPM as dot-lines (any smoothing
    is the caller's; say so in ``bpm_subtitle``). Zooming in the browser replaces
    any fixed zoom crop; plotly.js is inlined so the file works offline.
    """
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # No row-1 subplot title: it would sit exactly where the horizontal legend
    # lives (both land just above the top panel's domain) and overlap it.
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        row_heights=[0.52, 0.48],
                        subplot_titles=["", bpm_subtitle])
    for t, y, name, color in ((t_env, env, env_name, "firebrick"),
                              (t_act, act, act_name, "royalblue")):
        yy = np.asarray(y, dtype=float) / _robust_scale(y)
        td, yd = minmax_decimate(np.asarray(t, dtype=float), yy, max_points // 2)
        fig.add_trace(go.Scattergl(x=td, y=yd, name=name,
                                   line=dict(color=color, width=1)), row=1, col=1)
    span = np.array([0.0, 1.0])
    for beats, name, color, dash in ((beats_env, "NST beats", "firebrick", None),
                                     (beats_act, "FUNet beats", "royalblue", "dash")):
        bx, by = _beat_segments(beats, span)
        if bx.size:
            fig.add_trace(go.Scattergl(x=bx, y=by, mode="lines", name=name,
                                       hoverinfo="skip", opacity=0.55,
                                       line=dict(color=color, width=1, dash=dash)),
                          row=1, col=1)
    for t, y, name, color in ((t_bpm_env, bpm_env, bpm_env_name, "firebrick"),
                              (t_bpm_act, bpm_act, bpm_act_name, "royalblue")):
        fig.add_trace(go.Scattergl(x=np.asarray(t, dtype=float), y=np.asarray(y, dtype=float),
                                   mode="lines+markers", name=name, marker=dict(size=4),
                                   line=dict(color=color, width=1)), row=2, col=1)
    fig.update_layout(template="plotly_white", dragmode="pan",
                      height=height, hovermode="x unified", margin=dict(t=95),
                      title=dict(text=title, x=0, xanchor="left", y=0.985,
                                 yanchor="top", font=dict(size=14)),
                      legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom"))
    fig.update_xaxes(title_text=x_title, row=2, col=1,
                     rangeslider=dict(visible=True, thickness=0.06))
    fig.update_yaxes(title_text="scaled amplitude", row=1, col=1)
    fig.update_yaxes(title_text="bpm", row=2, col=1)
    # see save_interactive_multi: the range slider would otherwise lock y zoom
    fig.update_yaxes(fixedrange=False)
    fig.write_html(out_html, include_plotlyjs=True,
                   config={"scrollZoom": True, "displaylogo": False})
