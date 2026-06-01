"""
=============================================================================
  Cubic Crystal Structure Visualizer & Materials Science Calculator
  A production-ready interactive web app for Materials Science students.
=============================================================================
  Features:
  - SC, BCC, FCC, HCP crystal structure 3D visualization
  - Miller (hkl) plane cutting & shading
  - Interplanar Spacing Calculator (LaTeX step-by-step)
  - XRD Diffraction Pattern Simulator
  - Structure Factor Calculator (F_hkl)
  - Atomic Packing Factor (APF) Calculator
  - Coordination Number Visualization
=============================================================================
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Crystal Structure Explorer",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# GLOBAL PREMIUM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root & App ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #080c14; color: #cbd5e1; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f1520 0%, #0a0f1a 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }

/* ── Headings gradient ── */
h1 { font-family: 'Outfit', sans-serif; font-weight: 800;
     background: linear-gradient(135deg, #00f2fe 0%, #4facfe 50%, #a855f7 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     background-clip: text; font-size: 2.1rem !important; margin-bottom: 0 !important; }
h2 { font-family: 'Outfit', sans-serif; font-weight: 700;
     background: linear-gradient(90deg, #38bdf8 0%, #818cf8 100%);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     background-clip: text; font-size: 1.35rem !important; }
h3 { color: #94a3b8; font-size: 0.95rem !important; font-weight: 600;
     text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Tab bar ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: #0f172a;
    border-radius: 12px; padding: 4px;
    border: 1px solid #1e293b;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px; padding: 8px 18px;
    font-weight: 600; font-size: 0.88rem;
    color: #64748b; border: none;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #0ea5e9, #6366f1) !important;
    color: white !important;
}

/* ── Metric cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 18px; }
.kpi-card {
    background: linear-gradient(135deg, rgba(14,165,233,0.08) 0%, rgba(99,102,241,0.08) 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px; padding: 18px 16px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(99,102,241,0.55); }
.kpi-label { font-size: 0.72rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.1em; margin-bottom: 6px; }
.kpi-value { font-family: 'Outfit', sans-serif; font-size: 1.7rem;
             font-weight: 700; color: #00f2fe; line-height: 1.1; }
.kpi-unit  { font-size: 0.8rem; color: #475569; margin-top: 3px; }

/* ── Info callout ── */
.info-pill {
    background: rgba(14,165,233,0.1); border-left: 3px solid #0ea5e9;
    border-radius: 0 8px 8px 0; padding: 10px 14px;
    font-size: 0.88rem; color: #94a3b8; margin-bottom: 16px;
}

/* ── Section headers inside main ── */
.section-label {
    font-size: 0.78rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #475569; margin: 22px 0 6px;
}

/* ── Warning / Error ── */
.stAlert { border-radius: 10px; }

/* ── Number inputs / sliders ── */
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: #0f172a; color: #e2e8f0;
    border: 1px solid #1e293b; border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
#  SECTION 1 ─ CRYSTALLOGRAPHY MATH LIBRARY
# ════════════════════════════════════════════

def d_spacing(a, h, k, l, crystal_type="cubic", c=None):
    """Interplanar spacing for cubic (and HCP with c parameter)."""
    if h == 0 and k == 0 and l == 0:
        return None
    if crystal_type in ("SC", "BCC", "FCC"):
        return a / math.sqrt(h**2 + k**2 + l**2)
    else:  # HCP – hexagonal formula
        if c is None:
            c = a * 1.633
        denom = (4/3) * (h**2 + h*k + k**2) / (a**2) + l**2 / (c**2)
        return 1.0 / math.sqrt(denom) if denom > 0 else None


def atomic_radius(a, crystal_type, c=None):
    """Hard-sphere atomic radius."""
    if crystal_type == "SC":
        return a / 2.0
    elif crystal_type == "BCC":
        return math.sqrt(3) * a / 4.0
    elif crystal_type == "FCC":
        return math.sqrt(2) * a / 4.0
    else:  # HCP  (a is a-parameter)
        return a / 2.0


def close_packed_direction(crystal_type):
    dirs = {"SC": "[100]", "BCC": "[111]", "FCC": "[110]", "HCP": "[11̄20]"}
    return dirs.get(crystal_type, "—")


def coordination_number(crystal_type):
    cn = {"SC": 6, "BCC": 8, "FCC": 12, "HCP": 12}
    return cn.get(crystal_type, 0)


def packing_factor(crystal_type):
    """Atomic packing factor (APF)."""
    apfs = {
        "SC":  math.pi / 6,
        "BCC": math.pi * math.sqrt(3) / 8,
        "FCC": math.pi * math.sqrt(2) / 6,
        "HCP": math.pi * math.sqrt(2) / 6,   # same as FCC
    }
    return apfs.get(crystal_type, 0)


def atoms_per_unit_cell(crystal_type):
    n = {"SC": 1, "BCC": 2, "FCC": 4, "HCP": 6}
    return n.get(crystal_type, 0)


# ─── Structure factor F_hkl ───────────────────────────────────────────────
def structure_factor(crystal_type, h, k, l):
    """
    Returns |F_hkl|² (intensity-proportional) and a human-readable note.
    For an elemental crystal all atoms are identical (f_i = f).
    """
    def phase(x, y, z):
        return np.exp(2j * math.pi * (h*x + k*y + z*l))

    if crystal_type == "SC":
        atoms = [(0, 0, 0)]
    elif crystal_type == "BCC":
        atoms = [(0, 0, 0), (0.5, 0.5, 0.5)]
    elif crystal_type == "FCC":
        atoms = [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)]
    else:  # HCP (2-atom basis)
        atoms = [(0,0,0),(1/3, 2/3, 0.5)]

    F = sum(phase(x, y, z) for x, y, z in atoms)
    F_sq = abs(F)**2
    n_atoms = len(atoms)
    # Normalise relative to max (all atoms in phase)
    F_rel = F_sq / (n_atoms**2)

    if F_rel < 0.01:
        rule = "Forbidden (destructive interference) — intensity ≈ 0"
    elif F_rel > 0.99:
        rule = "Fully allowed (constructive interference) — maximum intensity"
    else:
        rule = f"Partially allowed — relative intensity {F_rel:.3f}"

    return F_sq, F_rel, rule, n_atoms


# ─── XRD peak generator ───────────────────────────────────────────────────
def xrd_peaks(crystal_type, a, wavelength_A, two_theta_max=90, c=None):
    """
    Generate (2θ, I_rel) XRD peaks for hkl up to |h|,|k|,|l| ≤ 6.
    Uses Bragg's law: 2d sinθ = λ  →  sinθ = λ/(2d).
    Intensities weighted by |F|² × multiplicity.
    """
    peak_dict = {}  # key = round(2theta,2), value = [I, label]

    if crystal_type == "HCP":
        hkl_range = range(-4, 5)
        l_range = range(-4, 5)
    else:
        hkl_range = range(-6, 7)
        l_range = range(-6, 7)

    for h in hkl_range:
        for k in hkl_range:
            for l in l_range:
                if h == 0 and k == 0 and l == 0:
                    continue
                d = d_spacing(a, h, k, l, crystal_type=crystal_type, c=c)
                if d is None or d <= 0:
                    continue
                sin_theta = wavelength_A / (2 * d)
                if not (0 < sin_theta <= 1):
                    continue
                theta = math.asin(sin_theta)
                two_theta = math.degrees(2 * theta)
                if two_theta > two_theta_max:
                    continue

                _, F_rel, _, _ = structure_factor(crystal_type, h, k, l)
                if F_rel < 0.005:
                    continue  # skip forbidden reflections

                key = round(two_theta, 1)
                if key in peak_dict:
                    peak_dict[key][0] += F_rel
                    peak_dict[key][1].add(f"({h}{k}{l})")
                else:
                    peak_dict[key] = [F_rel, {f"({h}{k}{l})"}]

    if not peak_dict:
        return [], []

    peaks = [(tt, v[0], ", ".join(sorted(v[1]))) for tt, v in peak_dict.items()]
    peaks.sort()
    max_I = max(p[1] for p in peaks)
    peaks = [(tt, I/max_I*100, lbl) for tt, I, lbl in peaks]
    return peaks


# ════════════════════════════════════════════
#  SECTION 2 ─ LATTICE POINT DEFINITIONS
# ════════════════════════════════════════════

def get_lattice_points(crystal_type, a=1.0, c=None):
    """
    Returns arrays of atom positions (fractional coords), categories, and
    neighbour bond pairs for coordination visualization.
    """
    corners = np.array([
        [0,0,0],[1,0,0],[0,1,0],[1,1,0],
        [0,0,1],[1,0,1],[0,1,1],[1,1,1],
    ], dtype=float)

    if crystal_type == "SC":
        pts = corners
        cats = ["Corner"] * 8
        bonds = []  # will build after

    elif crystal_type == "BCC":
        body = np.array([[0.5, 0.5, 0.5]])
        pts = np.vstack([corners, body])
        cats = ["Corner"] * 8 + ["Body Center"]
        bonds = []

    elif crystal_type == "FCC":
        faces = np.array([
            [0.5,0.5,0],[0.5,0.5,1],
            [0.5,0,0.5],[0.5,1,0.5],
            [0,0.5,0.5],[1,0.5,0.5],
        ])
        pts = np.vstack([corners, faces])
        cats = ["Corner"]*8 + ["Face Center"]*6
        bonds = []

    else:  # HCP  — show one unit cell (hexagonal prism)
        # Use 2-atom primitive basis plus periodic images
        c_ratio = (c if c else a*1.633) / a
        # Base layer (z=0): hexagon corners + center
        hcp_base = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, math.sqrt(3)/2, 0],
            [-0.5, math.sqrt(3)/2, 0],
            [-1, 0, 0],
            [-0.5, -math.sqrt(3)/2, 0],
            [0.5, -math.sqrt(3)/2, 0],
        ])
        # Top layer (z=c)
        hcp_top = hcp_base.copy()
        hcp_top[:, 2] = c_ratio
        # Mid layer (z=c/2) — tetrahedral sites
        hcp_mid = np.array([
            [0.5, math.sqrt(3)/6, c_ratio/2],
            [-0.5, math.sqrt(3)/6, c_ratio/2],
            [0, -math.sqrt(3)/3, c_ratio/2],
        ])
        pts = np.vstack([hcp_base, hcp_top, hcp_mid])
        cats = (["HCP Base"]*7 + ["HCP Top"]*7 + ["HCP Mid"]*3)
        bonds = []

    return pts, cats


# ════════════════════════════════════════════
#  SECTION 3 ─ PLANE GEOMETRY HELPERS
# ════════════════════════════════════════════

def plane_cube_intersections(h, k, l, C, eps=1e-7):
    """Intersect plane hx+ky+lz=C with the 12 edges of [0,1]³."""
    points = []
    for y in [0, 1]:
        for z in [0, 1]:
            if abs(h) > eps:
                x = (C - k*y - l*z) / h
                if -eps <= x <= 1+eps:
                    points.append((np.clip(x,0,1), float(y), float(z)))
    for x in [0, 1]:
        for z in [0, 1]:
            if abs(k) > eps:
                y = (C - h*x - l*z) / k
                if -eps <= y <= 1+eps:
                    points.append((float(x), np.clip(y,0,1), float(z)))
    for x in [0, 1]:
        for y in [0, 1]:
            if abs(l) > eps:
                z = (C - h*x - k*y) / l
                if -eps <= z <= 1+eps:
                    points.append((float(x), float(y), np.clip(z,0,1)))
    # Deduplicate
    unique = []
    for p in points:
        if not any(np.allclose(p, q, atol=1e-5) for q in unique):
            unique.append(p)
    return unique


def sort_polygon(points, h, k, l):
    """Sort coplanar 3-D points into convex polygon order."""
    if len(points) < 3:
        return points
    pts = np.array(points)
    c = pts.mean(0)
    n = np.array([h, k, l], dtype=float)
    n /= np.linalg.norm(n)
    ref = np.array([1,0,0]) if abs(n[0]) < 0.9 else np.array([0,1,0])
    u = np.cross(n, ref); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    angles = [math.atan2(np.dot(p-c, v), np.dot(p-c, u)) for p in pts]
    return pts[np.argsort(angles)].tolist()


def triangulate_polygon(ordered_pts):
    """Fan-triangulate a convex polygon. Returns xs,ys,zs,i,j,k."""
    if len(ordered_pts) < 3:
        return [], [], [], [], [], []
    xs = [p[0] for p in ordered_pts]
    ys = [p[1] for p in ordered_pts]
    zs = [p[2] for p in ordered_pts]
    N = len(xs)
    ii, jj, kk = [], [], []
    for idx in range(1, N-1):
        ii.append(0); jj.append(idx); kk.append(idx+1)
    return xs, ys, zs, ii, jj, kk


# ════════════════════════════════════════════
#  SECTION 4 ─ SIDEBAR CONTROLS
# ════════════════════════════════════════════

st.sidebar.markdown("## ⚛️ Crystal Explorer")
st.sidebar.markdown("---")

# ── Element Presets ──────────────────────────────────────
PRESETS = {
    "Custom":              {"type": "FCC", "a": 4.05, "c": None},
    "Aluminum (Al) – FCC": {"type": "FCC", "a": 4.05, "c": None},
    "Copper (Cu) – FCC":   {"type": "FCC", "a": 3.61, "c": None},
    "Gold (Au) – FCC":     {"type": "FCC", "a": 4.08, "c": None},
    "Iron α (Fe) – BCC":  {"type": "BCC", "a": 2.87, "c": None},
    "Chromium (Cr) – BCC": {"type": "BCC", "a": 2.88, "c": None},
    "Tungsten (W) – BCC":  {"type": "BCC", "a": 3.16, "c": None},
    "Polonium (Po) – SC":  {"type": "SC",  "a": 3.35, "c": None},
    "Titanium (Ti) – HCP": {"type": "HCP", "a": 2.95, "c": 4.68},
    "Magnesium (Mg) – HCP":{"type": "HCP", "a": 3.21, "c": 5.21},
    "Zinc (Zn) – HCP":     {"type": "HCP", "a": 2.67, "c": 4.95},
}

preset = st.sidebar.selectbox("🔬 Element Preset", list(PRESETS.keys()))
P = PRESETS[preset]
locked = (preset != "Custom")

st.sidebar.markdown("### Crystal Type")
TYPES = ["SC", "BCC", "FCC", "HCP"]
ctype = st.sidebar.radio("", TYPES,
    index=TYPES.index(P["type"]), horizontal=True,
    disabled=locked)
if locked:
    ctype = P["type"]

st.sidebar.markdown("### Lattice Parameters")
a_val = st.sidebar.number_input(
    "Lattice parameter  a  (Å)", min_value=0.5, max_value=20.0,
    value=float(P["a"]), step=0.01, format="%.3f", disabled=locked)

c_val = None
if ctype == "HCP":
    c_default = float(P["c"]) if P["c"] else round(a_val * 1.633, 3)
    c_val = st.sidebar.number_input(
        "Lattice parameter  c  (Å)", min_value=0.5, max_value=30.0,
        value=c_default, step=0.01, format="%.3f", disabled=locked)

st.sidebar.markdown("### Miller Indices  (hkl)")
hkl_cols = st.sidebar.columns(3)
with hkl_cols[0]: h = st.number_input("h", -5, 5, 1, 1)
with hkl_cols[1]: k = st.number_input("k", -5, 5, 1, 1)
with hkl_cols[2]: l = st.number_input("l", -5, 5, 0, 1)

st.sidebar.markdown("### Visualization Options")
# Plane shift slider
hkl_zero = (h == 0 and k == 0 and l == 0)
if not hkl_zero:
    cmin = float(sum(x for x in [h,k,l] if x < 0))
    cmax = float(sum(x for x in [h,k,l] if x > 0))
    if cmin == cmax:
        cmin -= 1; cmax += 1
    C_plane = st.sidebar.slider(
        "Plane shift  C  (hx+ky+lz = C)",
        cmin, cmax, float(round((cmin+cmax)/2, 1) if (cmin+cmax)/2 != 0 else 1.0),
        step=0.05)
else:
    C_plane = 1.0

atom_scale = st.sidebar.slider("Atom display size", 0.05, 0.55, 0.18, 0.01)

st.sidebar.markdown("### XRD Settings")
xrd_lambda = st.sidebar.number_input(
    "X-ray wavelength λ (Å)", min_value=0.5, max_value=3.0,
    value=1.5406, step=0.0001, format="%.4f",
    help="Cu Kα = 1.5406 Å (default)")


# ════════════════════════════════════════════
#  SECTION 5 ─ MAIN DASHBOARD
# ════════════════════════════════════════════

st.markdown("# ⚛️ Crystal Structure Explorer")
st.markdown(
    "<div class='info-pill'>Interactive materials science toolkit for "
    "crystal structure visualization, diffraction simulation, and "
    "structural analysis. Select a preset or customize parameters in the sidebar.</div>",
    unsafe_allow_html=True)

# ── Global validation ─────────────────────────────────────
if hkl_zero:
    st.error("⚠️ **Miller Indices cannot all be zero.** Adjust h, k, l in the sidebar.")
    st.stop()

# ── Pre-compute common quantities ────────────────────────
d = d_spacing(a_val, h, k, l, crystal_type=ctype, c=c_val)
r = atomic_radius(a_val, ctype, c=c_val)
apf = packing_factor(ctype)
cn = coordination_number(ctype)
cpd = close_packed_direction(ctype)
n_atoms = atoms_per_unit_cell(ctype)
F_sq, F_rel, F_rule, n_basis = structure_factor(ctype, h, k, l)

# ── KPI cards ─────────────────────────────────────────────
st.markdown("""
<div class='kpi-grid'>
  <div class='kpi-card'>
    <div class='kpi-label'>Interplanar Spacing d<sub>hkl</sub></div>
    <div class='kpi-value'>{d:.4f}</div>
    <div class='kpi-unit'>Angstroms (Å)</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Atomic Radius r</div>
    <div class='kpi-value'>{r:.4f}</div>
    <div class='kpi-unit'>Angstroms (Å)</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Packing Factor (APF)</div>
    <div class='kpi-value'>{apf:.4f}</div>
    <div class='kpi-unit'>{apf_pct:.1f}% of volume filled</div>
  </div>
</div>
<div class='kpi-grid'>
  <div class='kpi-card'>
    <div class='kpi-label'>Coordination Number</div>
    <div class='kpi-value' style='color:#a855f7'>{cn}</div>
    <div class='kpi-unit'>Nearest neighbours</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Atoms / Unit Cell</div>
    <div class='kpi-value' style='color:#f97316'>{n_atoms}</div>
    <div class='kpi-unit'>Effective atoms</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Close-Packed Direction</div>
    <div class='kpi-value' style='color:#10b981;font-size:1.4rem'>{cpd}</div>
    <div class='kpi-unit'>Max linear atomic density</div>
  </div>
</div>
""".format(d=d, r=r, apf=apf, apf_pct=apf*100, cn=cn, n_atoms=n_atoms, cpd=cpd),
unsafe_allow_html=True)

# ── Tabs ─────────────────────────────────────────────────
tabs = st.tabs([
    "🔮 3D Structure",
    "📐 Calculations",
    "📡 XRD Simulator",
    "⚡ Structure Factor",
    "📦 Packing & Coordination",
])


# ════════════════════════════════════════════════════════
#  TAB 1 ─ 3D CRYSTAL VISUALIZATION
# ════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 3D Unit Cell + Miller Plane")
    st.caption(f"Crystal type: **{ctype}** · a = {a_val} Å"
               + (f" · c = {c_val:.3f} Å" if c_val else "")
               + f" · Plane: **({h} {k} {l})** · C = {C_plane:.2f}")

    fig3d = go.Figure()

    # ─ Unit cell wireframe ─
    if ctype != "HCP":
        # Standard cubic box
        def cube_edges():
            ex, ey, ez = [], [], []
            for yv in [0,1]:
                for zv in [0,1]:
                    ex += [0, 1, None]; ey += [yv,yv,None]; ez += [zv,zv,None]
            for xv in [0,1]:
                for zv in [0,1]:
                    ex += [xv,xv,None]; ey += [0,1,None]; ez += [zv,zv,None]
            for xv in [0,1]:
                for yv in [0,1]:
                    ex += [xv,xv,None]; ey += [yv,yv,None]; ez += [0,1,None]
            return ex, ey, ez
        ex, ey, ez = cube_edges()
        fig3d.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='#334155', width=3),
            hoverinfo='none', name='Unit Cell', showlegend=True))
    else:
        # Hexagonal prism wireframe
        c_r = (c_val or a_val*1.633) / a_val
        angles_hex = [math.radians(i*60) for i in range(6)]
        hex_x = [math.cos(a) for a in angles_hex]
        hex_y = [math.sin(a) for a in angles_hex]
        ex, ey, ez = [], [], []
        # Base hexagon
        for i in range(6):
            ex += [hex_x[i], hex_x[(i+1)%6], None]
            ey += [hex_y[i], hex_y[(i+1)%6], None]
            ez += [0, 0, None]
        # Top hexagon
        for i in range(6):
            ex += [hex_x[i], hex_x[(i+1)%6], None]
            ey += [hex_y[i], hex_y[(i+1)%6], None]
            ez += [c_r, c_r, None]
        # Verticals
        for i in range(6):
            ex += [hex_x[i], hex_x[i], None]
            ey += [hex_y[i], hex_y[i], None]
            ez += [0, c_r, None]
        fig3d.add_trace(go.Scatter3d(
            x=ex, y=ey, z=ez, mode='lines',
            line=dict(color='#334155', width=3),
            hoverinfo='none', name='HCP Prism'))

    # ─ Atom spheres ─
    COLOR_MAP = {
        "Corner":      ("#94a3b8", "Corner Atoms"),
        "Body Center": ("#f97316", "Body-Center Atom"),
        "Face Center": ("#06b6d4", "Face-Center Atoms"),
        "HCP Base":    ("#94a3b8", "HCP Base"),
        "HCP Top":     ("#38bdf8", "HCP Top"),
        "HCP Mid":     ("#f59e0b", "HCP Mid (Tetrahedral)"),
    }
    pts, cats = get_lattice_points(ctype, a_val, c_val)
    msize = atom_scale * 130

    for cat_key, (col, label) in COLOR_MAP.items():
        mask = [c == cat_key for c in cats]
        sel = pts[mask]
        if len(sel) == 0:
            continue
        fig3d.add_trace(go.Scatter3d(
            x=sel[:,0], y=sel[:,1], z=sel[:,2],
            mode='markers',
            marker=dict(size=msize, color=col, opacity=0.92,
                        line=dict(color='#0f172a', width=1.5)),
            name=label,
            hovertemplate=f"<b>{label}</b><br>x=%{{x:.3f}}, y=%{{y:.3f}}, z=%{{z:.3f}}<extra></extra>"))

    # ─ Miller plane (cubic only for now) ─
    if ctype != "HCP":
        ipts = plane_cube_intersections(h, k, l, C_plane)
        if len(ipts) >= 3:
            opoly = sort_polygon(ipts, h, k, l)
            xs, ys, zs, ii, jj, kk = triangulate_polygon(opoly)
            if ii:
                fig3d.add_trace(go.Mesh3d(
                    x=xs, y=ys, z=zs, i=ii, j=jj, k=kk,
                    color='#6366f1', opacity=0.40,
                    flatshading=True,
                    name=f'({h}{k}{l}) Plane', showlegend=True))
                bx = xs + [xs[0]]; by = ys + [ys[0]]; bz = zs + [zs[0]]
                fig3d.add_trace(go.Scatter3d(
                    x=bx, y=by, z=bz, mode='lines',
                    line=dict(color='#a5f3fc', width=5),
                    hoverinfo='none', name='Plane Edge'))
        else:
            st.info(f"Plane ({h}{k}{l}) at C={C_plane:.2f} doesn't intersect the cell. Adjust the Plane shift slider.")

    # ─ Layout ─
    tick_v = [0, 0.5, 1]
    tick_t = ["0", f"{a_val/2:.2f}", f"{a_val:.2f} Å"]
    fig3d.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=30),
        height=580,
        legend=dict(
            bgcolor='rgba(10,15,25,0.85)',
            bordercolor='#1e293b', borderwidth=1,
            font=dict(size=11)),
        scene=dict(
            xaxis=dict(title='X', tickvals=tick_v, ticktext=tick_t,
                       gridcolor='#1e293b', zerolinecolor='#334155'),
            yaxis=dict(title='Y', tickvals=tick_v, ticktext=tick_t,
                       gridcolor='#1e293b', zerolinecolor='#334155'),
            zaxis=dict(title='Z', tickvals=tick_v, ticktext=tick_t,
                       gridcolor='#1e293b', zerolinecolor='#334155'),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.6, y=1.4, z=1.2))))
    st.plotly_chart(fig3d, use_container_width=True)


# ════════════════════════════════════════════════════════
#  TAB 2 ─ ANALYTICAL CALCULATIONS (LaTeX steps)
# ════════════════════════════════════════════════════════
with tabs[1]:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("## Interplanar Spacing")
        with st.expander("Step-by-step derivation", expanded=True):
            if ctype in ("SC","BCC","FCC"):
                st.write("**Cubic system formula:**")
                st.latex(r"d_{hkl} = \frac{a}{\sqrt{h^2+k^2+l^2}}")
                st.write("Substituting values:")
                st.latex(rf"d_{{{h}{k}{l}}} = \frac{{{a_val:.3f}}}{{\sqrt{{({h})^2+({k})^2+({l})^2}}}}")
                hkl2 = h**2+k**2+l**2
                st.latex(rf"= \frac{{{a_val:.3f}}}{{\sqrt{{{hkl2}}}}} = \frac{{{a_val:.3f}}}{{{math.sqrt(hkl2):.4f}}}")
                st.latex(rf"\boxed{{d_{{{h}{k}{l}}} = {d:.4f}\ \text{{Å}}}}")
            else:
                cv = c_val or a_val*1.633
                st.write("**Hexagonal (HCP) system formula:**")
                st.latex(r"\frac{1}{d^2} = \frac{4}{3}\cdot\frac{h^2+hk+k^2}{a^2} + \frac{l^2}{c^2}")
                denom = (4/3)*(h**2+h*k+k**2)/(a_val**2) + l**2/(cv**2)
                st.write("Substituting values:")
                st.latex(
                    rf"\frac{{1}}{{d^2}} = \frac{{4}}{{3}}\cdot"
                    rf"\frac{{{h}^2+({h})({k})+{k}^2}}{{{a_val:.3f}^2}} + "
                    rf"\frac{{{l}^2}}{{{cv:.3f}^2}} = {denom:.6f}")
                st.latex(rf"\boxed{{d_{{{h}{k}{l}}} = {d:.4f}\ \text{{Å}}}}")

        st.markdown("## Atomic Radius")
        with st.expander("Radius derivation", expanded=True):
            if ctype == "SC":
                st.latex(r"2r = a \Rightarrow r = \frac{a}{2}")
                st.latex(rf"r = \frac{{{a_val:.3f}}}{{2}} = {r:.4f}\ \text{{Å}}")
            elif ctype == "BCC":
                st.latex(r"4r = \sqrt{3}\,a \Rightarrow r = \frac{\sqrt{3}\,a}{4}")
                st.latex(rf"r = \frac{{\sqrt{{3}}\times{a_val:.3f}}}{{4}} = {r:.4f}\ \text{{Å}}")
            elif ctype == "FCC":
                st.latex(r"4r = \sqrt{2}\,a \Rightarrow r = \frac{\sqrt{2}\,a}{4}")
                st.latex(rf"r = \frac{{\sqrt{{2}}\times{a_val:.3f}}}{{4}} = {r:.4f}\ \text{{Å}}")
            else:
                st.latex(r"2r = a \Rightarrow r = \frac{a}{2}")
                st.latex(rf"r = \frac{{{a_val:.3f}}}{{2}} = {r:.4f}\ \text{{Å}}")

    with col_b:
        st.markdown("## Crystal Type Summary")
        info_table = {
            "Crystal Type": ctype,
            "Lattice Parameter a": f"{a_val:.3f} Å",
            "c parameter": f"{c_val:.3f} Å" if c_val else "N/A",
            "Atoms per Unit Cell": str(n_atoms),
            "Atomic Radius": f"{r:.4f} Å",
            "Coordination Number": str(cn),
            "Close-Packed Direction": cpd,
            "APF": f"{apf:.4f}  ({apf*100:.1f}%)",
        }
        for k_i, v_i in info_table.items():
            st.markdown(
                f"<div style='display:flex;justify-content:space-between;"
                f"padding:8px 12px;border-bottom:1px solid #1e293b'>"
                f"<span style='color:#64748b;font-size:0.88rem'>{k_i}</span>"
                f"<span style='color:#e2e8f0;font-weight:600'>{v_i}</span>"
                f"</div>", unsafe_allow_html=True)

        st.markdown("## Bragg's Law")
        with st.expander("2d sinθ = nλ", expanded=True):
            st.latex(r"2\,d_{hkl}\sin\theta = n\lambda")
            if d and d > 0:
                sinT = xrd_lambda / (2*d)
                if 0 < sinT <= 1:
                    theta_deg = math.degrees(math.asin(sinT))
                    two_theta = 2 * theta_deg
                    st.write(f"For n=1, λ = {xrd_lambda:.4f} Å, d = {d:.4f} Å:")
                    st.latex(rf"\sin\theta = \frac{{\lambda}}{{2d}} = \frac{{{xrd_lambda:.4f}}}{{2\times{d:.4f}}} = {sinT:.4f}")
                    st.latex(rf"\theta = {theta_deg:.2f}°\quad\Rightarrow\quad 2\theta = {two_theta:.2f}°")
                    st.success(f"✅ XRD Peak expected at  **2θ = {two_theta:.2f}°**")
                else:
                    st.warning("No diffraction peak for this d-spacing with the selected wavelength (sinθ > 1).")


# ════════════════════════════════════════════════════════
#  TAB 3 ─ XRD DIFFRACTION PATTERN SIMULATOR
# ════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("## XRD Diffraction Pattern Simulator")
    st.caption(
        f"Crystal: **{ctype}** · a = {a_val} Å · λ = {xrd_lambda:.4f} Å (Cu Kα = 1.5406 Å)")

    col_xrd1, col_xrd2 = st.columns([3,1])
    with col_xrd2:
        two_theta_max = st.slider("Max 2θ (°)", 30, 130, 90, 5)
        show_labels = st.checkbox("Show hkl labels", True)
        peak_width = st.slider("Peak width (°)", 0.1, 2.0, 0.4, 0.05)

    with col_xrd1:
        peaks = xrd_peaks(ctype, a_val, xrd_lambda, two_theta_max=two_theta_max, c=c_val)

        if not peaks:
            st.warning("No diffraction peaks found for these parameters. Try increasing max 2θ or using a longer wavelength.")
        else:
            # Build a simulated pattern (sum of Gaussians)
            tt_arr = np.linspace(5, two_theta_max, 2000)
            I_arr = np.zeros_like(tt_arr)
            sigma = peak_width / 2.355  # FWHM → sigma
            for tt, I_rel, _ in peaks:
                I_arr += I_rel * np.exp(-0.5*((tt_arr - tt)/sigma)**2)

            fig_xrd = go.Figure()
            # Continuous pattern
            fig_xrd.add_trace(go.Scatter(
                x=tt_arr, y=I_arr, mode='lines',
                line=dict(color='#06b6d4', width=1.8),
                fill='tozeroy', fillcolor='rgba(6,182,212,0.12)',
                name='Simulated pattern'))

            # Stem markers at peak positions
            for tt, I_rel, lbl in peaks:
                fig_xrd.add_trace(go.Scatter(
                    x=[tt, tt, None], y=[0, I_rel, None],
                    mode='lines', line=dict(color='#f97316', width=1.5),
                    showlegend=False, hoverinfo='none'))
                if show_labels:
                    fig_xrd.add_annotation(
                        x=tt, y=I_rel + 2,
                        text=lbl.split(",")[0],
                        font=dict(size=8, color='#94a3b8'),
                        showarrow=False, textangle=-60)

            fig_xrd.update_layout(
                template='plotly_dark',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(8,12,20,1)',
                xaxis=dict(title='2θ (°)', gridcolor='#1e293b',
                           range=[5, two_theta_max]),
                yaxis=dict(title='Relative Intensity (%)',
                           gridcolor='#1e293b', range=[0, 110]),
                margin=dict(l=60, r=20, b=60, t=30),
                height=420,
                showlegend=False)
            st.plotly_chart(fig_xrd, use_container_width=True)

            # Peak table
            st.markdown("### 📋 Peak List")
            header_cols = st.columns([1,1,1,2])
            header_cols[0].markdown("**2θ (°)**")
            header_cols[1].markdown("**I_rel (%)**")
            header_cols[2].markdown("**d (Å)**")
            header_cols[3].markdown("**hkl families**")
            for tt, I_rel, lbl in sorted(peaks, key=lambda x: x[0]):
                sinT = math.sin(math.radians(tt/2))
                d_meas = xrd_lambda / (2*sinT) if sinT > 0 else 0
                pc = st.columns([1,1,1,2])
                pc[0].write(f"{tt:.2f}")
                pc[1].write(f"{I_rel:.1f}")
                pc[2].write(f"{d_meas:.4f}")
                pc[3].write(lbl)


# ════════════════════════════════════════════════════════
#  TAB 4 ─ STRUCTURE FACTOR CALCULATOR
# ════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("## Structure Factor  F(hkl)")
    st.markdown(
        "The structure factor determines whether a reflection is **allowed** "
        "or **forbidden** by symmetry (systematic absences).")

    col_sf1, col_sf2 = st.columns(2)

    with col_sf1:
        st.markdown("### Formula")
        st.latex(r"F_{hkl} = \sum_{j=1}^{N} f_j \, e^{2\pi i (hx_j + ky_j + lz_j)}")
        st.write("For an elemental crystal all scattering factors fⱼ = f:")
        st.latex(r"F_{hkl} = f \sum_{j} e^{2\pi i (hx_j + ky_j + lz_j)}")
        st.write("and intensity is proportional to |F|².")

        # Show basis atoms
        st.markdown("### Atom Basis")
        bases = {
            "SC":  [(0,0,0)],
            "BCC": [(0,0,0),(0.5,0.5,0.5)],
            "FCC": [(0,0,0),(0.5,0.5,0),(0.5,0,0.5),(0,0.5,0.5)],
            "HCP": [(0,0,0),(1/3,2/3,0.5)],
        }
        for i,(x,y,z) in enumerate(bases[ctype]):
            st.markdown(
                f"<div style='padding:6px 12px;border-left:3px solid #6366f1;"
                f"margin:4px 0;font-family:JetBrains Mono,monospace;font-size:0.85rem'>"
                f"Atom {i+1}:  ({x:.4f}, {y:.4f}, {z:.4f})</div>",
                unsafe_allow_html=True)

    with col_sf2:
        st.markdown(f"### Result for  ({h} {k} {l})")

        # Show phase calculation step by step
        def phase_str(x, y, z):
            exp_arg = 2*(h*x + k*y + l*z)
            return f"e^{{i\\pi({h}·{x:.4f}+{k}·{y:.4f}+{l}·{z:.4f})·2}} = e^{{i\\pi·{exp_arg:.4f}}}"

        for i,(x,y,z) in enumerate(bases[ctype]):
            st.latex(rf"\text{{Atom {i+1}:}} \quad {phase_str(x,y,z)}")

        st.latex(rf"|F|^2 = {F_sq:.4f} \cdot f^2")
        st.latex(rf"\text{{Relative intensity}} = \frac{{|F|^2}}{{N^2 f^2}} = {F_rel:.4f}")

        if F_rel < 0.01:
            st.error(f"🚫 **FORBIDDEN** reflection  —  systematic absence\n\n{F_rule}")
        elif F_rel > 0.99:
            st.success(f"✅ **FULLY ALLOWED** reflection\n\n{F_rule}")
        else:
            st.warning(f"⚠️ **PARTIALLY ALLOWED**\n\n{F_rule}")

        # Systematic absence rules table
        st.markdown("### Systematic Absence Rules")
        rules = {
            "SC":  "No systematic absences — all (hkl) allowed",
            "BCC": "Forbidden when h+k+l = **odd**",
            "FCC": "Forbidden when hkl are **mixed** (not all odd or all even)",
            "HCP": "Forbidden when h+2k = 3n and l = **odd**",
        }
        st.info(f"**{ctype}:**  {rules[ctype]}")

        # Build a mini grid showing allowed reflections
        st.markdown("### Allowed Reflections (low index)")
        common = [(1,0,0),(1,1,0),(1,1,1),(2,0,0),(2,1,0),(2,1,1),
                  (2,2,0),(3,0,0),(3,1,0),(3,1,1),(2,2,2)]
        grid_cols = st.columns(4)
        for ci,(hi,ki,li) in enumerate(common):
            _,fr,_,_ = structure_factor(ctype, hi, ki, li)
            allowed = fr > 0.01
            color = "#10b981" if allowed else "#ef4444"
            symbol = "✓" if allowed else "✗"
            grid_cols[ci%4].markdown(
                f"<div style='text-align:center;padding:6px;margin:3px;"
                f"border-radius:6px;border:1px solid {color}20;"
                f"background:{color}15;font-size:0.8rem;color:{color}'>"
                f"{symbol} ({hi}{ki}{li})</div>", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════
#  TAB 5 ─ PACKING FACTOR & COORDINATION NUMBER
# ════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("## Atomic Packing Factor & Coordination Number")
    col_p1, col_p2 = st.columns(2)

    with col_p1:
        st.markdown("### APF Derivation")
        st.latex(r"\text{APF} = \frac{N_{\text{atoms}} \cdot \tfrac{4}{3}\pi r^3}{V_{\text{cell}}}")

        if ctype == "SC":
            st.write("**SC:** 1 atom, r = a/2, V_cell = a³")
            st.latex(r"\text{APF} = \frac{1 \cdot \frac{4}{3}\pi\left(\frac{a}{2}\right)^3}{a^3} = \frac{\pi}{6} \approx 0.5236")
        elif ctype == "BCC":
            st.write("**BCC:** 2 atoms, r = √3a/4, V_cell = a³")
            st.latex(r"\text{APF} = \frac{2 \cdot \frac{4}{3}\pi\left(\frac{\sqrt{3}a}{4}\right)^3}{a^3} = \frac{\pi\sqrt{3}}{8} \approx 0.6802")
        elif ctype == "FCC":
            st.write("**FCC:** 4 atoms, r = √2a/4, V_cell = a³")
            st.latex(r"\text{APF} = \frac{4 \cdot \frac{4}{3}\pi\left(\frac{\sqrt{2}a}{4}\right)^3}{a^3} = \frac{\pi\sqrt{2}}{6} \approx 0.7405")
        else:
            st.write("**HCP:** 6 atoms per cell, same APF as FCC")
            st.latex(r"\text{APF} = \frac{\pi\sqrt{2}}{6} \approx 0.7405")

        st.latex(rf"\text{{APF}}_{{ {ctype} }} = {apf:.4f} \quad ({apf*100:.2f}\%)")

        # APF comparison bar chart
        st.markdown("### APF Comparison")
        structures = ["SC", "BCC", "FCC", "HCP"]
        apf_vals = [packing_factor(s)*100 for s in structures]
        colors = ['#64748b','#f97316','#06b6d4','#a855f7']
        highlight = ['#ef4444' if s == ctype else c for s, c in zip(structures, colors)]

        fig_apf = go.Figure(go.Bar(
            x=structures, y=apf_vals,
            marker=dict(color=highlight, line=dict(width=0)),
            text=[f"{v:.1f}%" for v in apf_vals],
            textposition='outside',
            textfont=dict(color='#cbd5e1', size=13)))
        fig_apf.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(8,12,20,1)',
            yaxis=dict(title='APF (%)', range=[0, 85],
                       gridcolor='#1e293b'),
            xaxis=dict(gridcolor='#1e293b'),
            margin=dict(l=50, r=20, b=40, t=40),
            height=300,
            showlegend=False)
        fig_apf.add_hline(y=apf*100, line=dict(color='#ef4444', dash='dash'),
                          annotation_text=f"Current: {ctype} ({apf*100:.1f}%)",
                          annotation_position="top right")
        st.plotly_chart(fig_apf, use_container_width=True)

    with col_p2:
        st.markdown("### Coordination Number Visualization")
        st.markdown(f"**{ctype}** has a coordination number of **{cn}**.")

        cn_descriptions = {
            "SC":  "6 nearest neighbours — one along each face of the cube: ±x, ±y, ±z.",
            "BCC": "8 nearest neighbours — the 8 corner atoms surrounding the body-center (or vice versa).",
            "FCC": "12 nearest neighbours — 4 in the same plane, 4 above, 4 below.",
            "HCP": "12 nearest neighbours — 6 in-plane, 3 above, 3 below (same as FCC).",
        }
        st.info(cn_descriptions[ctype])

        # 3D coordination figure
        fig_cn = go.Figure()

        # Centre atom (red/highlighted)
        fig_cn.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0], mode='markers',
            marker=dict(size=22, color='#ef4444',
                        line=dict(color='#fca5a5', width=3)),
            name='Central Atom'))

        # Neighbour positions
        cn_neighbours = {
            "SC":  [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)],
            "BCC": [(1,1,1),(1,1,-1),(1,-1,1),(-1,1,1),
                    (1,-1,-1),(-1,1,-1),(-1,-1,1),(-1,-1,-1)],
            "FCC": [(1,1,0),(1,-1,0),(-1,1,0),(-1,-1,0),
                    (1,0,1),(1,0,-1),(-1,0,1),(-1,0,-1),
                    (0,1,1),(0,1,-1),(0,-1,1),(0,-1,-1)],
            "HCP": [(1,0,0),(-1,0,0),(0.5,math.sqrt(3)/2,0),(-0.5,math.sqrt(3)/2,0),
                    (0.5,-math.sqrt(3)/2,0),(-0.5,-math.sqrt(3)/2,0),
                    (0.5,math.sqrt(3)/6,math.sqrt(2/3)),(-0.5,math.sqrt(3)/6,math.sqrt(2/3)),
                    (0,-math.sqrt(3)/3,math.sqrt(2/3)),
                    (0.5,math.sqrt(3)/6,-math.sqrt(2/3)),(-0.5,math.sqrt(3)/6,-math.sqrt(2/3)),
                    (0,-math.sqrt(3)/3,-math.sqrt(2/3))],
        }

        nbrs = cn_neighbours.get(ctype, [])
        nx = [n[0] for n in nbrs]
        ny = [n[1] for n in nbrs]
        nz = [n[2] for n in nbrs]

        # Bond lines
        bond_x, bond_y, bond_z = [], [], []
        for bx_, by_, bz_ in nbrs:
            bond_x += [0, bx_, None]
            bond_y += [0, by_, None]
            bond_z += [0, bz_, None]
        fig_cn.add_trace(go.Scatter3d(
            x=bond_x, y=bond_y, z=bond_z, mode='lines',
            line=dict(color='#334155', width=3),
            hoverinfo='none', showlegend=False))

        # Neighbour atoms
        fig_cn.add_trace(go.Scatter3d(
            x=nx, y=ny, z=nz, mode='markers',
            marker=dict(size=16, color='#06b6d4',
                        line=dict(color='#a5f3fc', width=2)),
            name=f'{cn} Nearest Neighbours'))

        # Number labels
        for i,(bx_,by_,bz_) in enumerate(nbrs):
            fig_cn.add_trace(go.Scatter3d(
                x=[bx_*1.08], y=[by_*1.08], z=[bz_*1.08],
                mode='text', text=[str(i+1)],
                textfont=dict(size=9, color='#94a3b8'),
                showlegend=False, hoverinfo='none'))

        fig_cn.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=0, b=0, t=10),
            height=400,
            legend=dict(bgcolor='rgba(10,15,25,0.85)',
                        bordercolor='#1e293b', borderwidth=1),
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='cube',
                camera=dict(eye=dict(x=1.8, y=1.4, z=1.2))))
        st.plotly_chart(fig_cn, use_container_width=True)

        # Summary table – all structures
        st.markdown("### All Structures Summary")
        summary_data = [
            ("SC",  6,  1, f"{packing_factor('SC')*100:.1f}%", "[100]"),
            ("BCC", 8,  2, f"{packing_factor('BCC')*100:.1f}%", "[111]"),
            ("FCC", 12, 4, f"{packing_factor('FCC')*100:.1f}%", "[110]"),
            ("HCP", 12, 6, f"{packing_factor('HCP')*100:.1f}%", "[11̄20]"),
        ]
        cols = st.columns([1,1,1,1,2])
        for hdr, w in zip(["Type","CN","N/cell","APF","CP Dir"],[1,1,1,1,2]):
            _ = None  # just labelling
        header = st.columns([1,1,1,1,2])
        for col_i, lbl in zip(header, ["Type","CN","N/cell","APF","CP Dir"]):
            col_i.markdown(f"**{lbl}**")
        for row in summary_data:
            highlight_row = row[0] == ctype
            bg = "rgba(99,102,241,0.15)" if highlight_row else "transparent"
            rc = st.columns([1,1,1,1,2])
            for col_i, val in zip(rc, row):
                col_i.markdown(
                    f"<div style='background:{bg};padding:5px 4px;"
                    f"border-radius:4px;font-size:0.9rem'>{val}</div>",
                    unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#334155;font-size:0.8rem;padding:10px'>"
    "⚛️ Crystal Structure Explorer — Built for Materials Science Students | "
    "SC · BCC · FCC · HCP · XRD · Structure Factor · Packing Factor · Coordination Number"
    "</div>", unsafe_allow_html=True)
