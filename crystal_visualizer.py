import streamlit as st
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. PAGE SETUP & PREMIUM CUSTOM STYLING
# ==========================================
st.set_page_config(
    page_title="Cubic Crystal Visualizer & Spacing Calculator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern glassmorphism, dark theme, and high-fidelity typography
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;800&family=Outfit:wght@400;600;800&display=swap');
    
    /* Main Layout Styling */
    .stApp {
        background-color: #0d0f13;
        color: #e2e8f0;
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, .title-text {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #161a22;
        border-right: 1px solid #2d3748;
    }
    
    /* Cards / Containers for Analytical Calculations */
    .calc-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease, border-color 0.3s ease;
    }
    .calc-card:hover {
        transform: translateY(-2px);
        border-color: rgba(0, 242, 254, 0.3);
    }
    
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 700;
        color: #00f2fe;
        margin: 5px 0;
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Responsive margins */
    .main-header {
        margin-bottom: 30px;
    }
    
    /* Instructions */
    .info-box {
        background: rgba(14, 165, 233, 0.1);
        border-left: 4px solid #0ea5e9;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 25px;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. HELPER FUNCTIONS FOR CALCULATIONS
# ==========================================
def calculate_d_spacing(a, h, k, l):
    """Calculates interplanar spacing d for cubic structure."""
    if h == 0 and k == 0 and l == 0:
        return None
    return a / np.sqrt(h**2 + k**2 + l**2)

def get_crystal_properties(crystal_type, a):
    """Returns atomic radius and close-packed direction."""
    if crystal_type == "Simple Cubic (SC)":
        r = a / 2.0
        direction = "[100]"
        packing_relation = r"a = 2r \Rightarrow r = \frac{a}{2}"
    elif crystal_type == "Body-Centered Cubic (BCC)":
        r = (np.sqrt(3) / 4.0) * a
        direction = "[111]"
        packing_relation = r"4r = \sqrt{3}a \Rightarrow r = \frac{\sqrt{3}}{4}a"
    else:  # FCC
        r = (np.sqrt(2) / 4.0) * a
        direction = "[110]"
        packing_relation = r"4r = \sqrt{2}a \Rightarrow r = \frac{\sqrt{2}}{4}a"
    
    return r, direction, packing_relation

# ==========================================
# 3. 3D VISUALIZATION ENGINE (PLOTLY)
# ==========================================
def get_lattice_points(crystal_type):
    """Returns fractional coordinates and labels/categories of atoms for cubic unit cell."""
    corners = [
        [0,0,0], [1,0,0], [0,1,0], [1,1,0],
        [0,0,1], [1,0,1], [0,1,1], [1,1,1]
    ]
    
    if crystal_type == "Simple Cubic (SC)":
        # Only corner atoms
        return np.array(corners), ["Corner"] * 8
        
    elif crystal_type == "Body-Centered Cubic (BCC)":
        # Corners + Body center
        pts = corners + [[0.5, 0.5, 0.5]]
        categories = ["Corner"] * 8 + ["Body Center"]
        return np.array(pts), categories
        
    else:  # Face-Centered Cubic (FCC)
        # Corners + Face centers
        face_centers = [
            [0.5, 0.5, 0], [0.5, 0.5, 1],  # XY faces
            [0.5, 0, 0.5], [0.5, 1, 0.5],  # XZ faces
            [0, 0.5, 0.5], [1, 0.5, 0.5]   # YZ faces
        ]
        pts = corners + face_centers
        categories = ["Corner"] * 8 + ["Face Center"] * 6
        return np.array(pts), categories

def get_plane_intersections(h, k, l, C):
    """
    Finds the intersection points of the plane h*x + k*y + l*z = C 
    with the 12 edges of the unit cell cube [0, 1]^3.
    """
    points = []
    eps = 1e-7
    
    # Edges parallel to X: y in {0,1}, z in {0,1}, x in [0,1]
    if abs(h) > eps:
        for y in [0, 1]:
            for z in [0, 1]:
                x = (C - k * y - l * z) / h
                if 0.0 - eps <= x <= 1.0 + eps:
                    points.append((max(0.0, min(1.0, x)), y, z))
                    
    # Edges parallel to Y: x in {0,1}, z in {0,1}, y in [0,1]
    if abs(k) > eps:
        for x in [0, 1]:
            for z in [0, 1]:
                y = (C - h * x - l * z) / k
                if 0.0 - eps <= y <= 1.0 + eps:
                    points.append((x, max(0.0, min(1.0, y)), z))
                    
    # Edges parallel to Z: x in {0,1}, y in {0,1}, z in [0,1]
    if abs(l) > eps:
        for x in [0, 1]:
            for y in [0, 1]:
                z = (C - h * x - k * y) / l
                if 0.0 - eps <= z <= 1.0 + eps:
                    points.append((x, y, max(0.0, min(1.0, z))))
                    
    # Remove duplicate points
    unique_points = []
    for p in points:
        duplicate = False
        for up in unique_points:
            if np.allclose(p, up, atol=1e-5):
                duplicate = True
                break
        if not duplicate:
            unique_points.append(p)
            
    return unique_points

def order_polygon_vertices(points, h, k, l):
    """Orders coplanar 3D points in a convex polygon contour relative to their center of mass."""
    if len(points) < 3:
        return points
        
    pts = np.array(points)
    center = np.mean(pts, axis=0)
    
    n = np.array([h, k, l], dtype=float)
    n_norm = n / np.linalg.norm(n)
    
    # Define coordinate system on the plane
    if abs(n_norm[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])
        
    u = np.cross(n_norm, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n_norm, u)
    
    # Calculate angles
    angles = []
    for p in pts:
        diff = p - center
        x_proj = np.dot(diff, u)
        y_proj = np.dot(diff, v)
        angles.append(np.arctan2(y_proj, x_proj))
        
    sorted_indices = np.argsort(angles)
    return pts[sorted_indices].tolist()

def get_polygon_mesh_data(ordered_points):
    """
    Triangulates the convex polygon vertices ordered_points to create mesh data.
    Returns: xs, ys, zs, i_idx, j_idx, k_idx
    """
    N = len(ordered_points)
    if N < 3:
        return [], [], [], [], [], []
    
    xs = [p[0] for p in ordered_points]
    ys = [p[1] for p in ordered_points]
    zs = [p[2] for p in ordered_points]
    
    # Triangulate: connect vertex 0 to all other vertices (0, idx, idx+1)
    i_idx = []
    j_idx = []
    k_idx = []
    for idx in range(1, N - 1):
        i_idx.append(0)
        j_idx.append(idx)
        k_idx.append(idx + 1)
        
    return xs, ys, zs, i_idx, j_idx, k_idx

# ==========================================
# 4. ELEMENT PRESETS & SIDEBAR
# ==========================================
st.sidebar.markdown("<h2 style='margin-top: 0;'>⚙️ Controls</h2>", unsafe_allow_html=True)

# Presets selector
presets = {
    "Custom (Manual Input)": {"type": "Face-Centered Cubic (FCC)", "a": 4.05, "desc": "Custom Parameters"},
    "Aluminum (Al) - FCC": {"type": "Face-Centered Cubic (FCC)", "a": 4.05, "desc": "Standard FCC metal with a = 4.05 Å"},
    "Copper (Cu) - FCC": {"type": "Face-Centered Cubic (FCC)", "a": 3.61, "desc": "High conductivity FCC metal with a = 3.61 Å"},
    "Iron (α-Fe) - BCC": {"type": "Body-Centered Cubic (BCC)", "a": 2.87, "desc": "Ferritic steel base at room temp with a = 2.87 Å"},
    "Chromium (Cr) - BCC": {"type": "Body-Centered Cubic (BCC)", "a": 2.88, "desc": "Corrosion resistant BCC metal with a = 2.88 Å"},
    "Polonium (Po) - SC": {"type": "Simple Cubic (SC)", "a": 3.42, "desc": "Rare simple cubic metal with a = 3.42 Å"},
    "Gold (Au) - FCC": {"type": "Face-Centered Cubic (FCC)", "a": 4.08, "desc": "Noble FCC metal with a = 4.08 Å"}
}

preset_choice = st.sidebar.selectbox("Select Element Preset", list(presets.keys()))
chosen_preset = presets[preset_choice]

# Crystal Type Selection
crystal_types = ["Simple Cubic (SC)", "Body-Centered Cubic (BCC)", "Face-Centered Cubic (FCC)"]
if preset_choice == "Custom (Manual Input)":
    crystal_type = st.sidebar.radio("Cubic Crystal Type", crystal_types, index=2)
    a = st.sidebar.number_input("Lattice Parameter 'a' (Å)", min_value=0.01, max_value=20.0, value=4.05, step=0.01, format="%.2f")
else:
    # Disable controls by selecting preset values
    st.sidebar.info(f"**Preset:** {chosen_preset['desc']}")
    crystal_type = st.sidebar.radio("Cubic Crystal Type", crystal_types, index=crystal_types.index(chosen_preset['type']), disabled=True)
    a = st.sidebar.number_input("Lattice Parameter 'a' (Å)", min_value=0.01, max_value=20.0, value=chosen_preset['a'], disabled=True, format="%.2f")

# Input fields for Miller Indices (h, k, l)
st.sidebar.markdown("### Miller Indices (hkl)")
col_h, col_k, col_l = st.sidebar.columns(3)
with col_h:
    h = st.number_input("h", min_value=-3, max_value=3, value=1, step=1)
with col_k:
    k = st.number_input("k", min_value=-3, max_value=3, value=1, step=1)
with col_l:
    l = st.number_input("l", min_value=-3, max_value=3, value=0, step=1)

# Plane positioning slider (C value)
V_min = sum([i for i in [h, k, l] if i < 0])
V_max = sum([i for i in [h, k, l] if i > 0])

# Determine a reasonable default for C
if h == 0 and k == 0 and l == 0:
    default_C = 0.0
else:
    if V_min < 1.0 <= V_max:
        default_C = 1.0
    elif V_min < 0.0 < V_max:
        default_C = 0.0
    else:
        default_C = float(V_min + V_max) / 2.0

st.sidebar.markdown("### Visualization Options")

# Interactive slider for shifting the plane
if h != 0 or k != 0 or l != 0:
    C = st.sidebar.slider(
        "Plane Position (C in hx + ky + lz = C)",
        float(V_min), float(V_max), float(default_C),
        step=0.05,
        help="Slide to shift the parallel planes through the unit cell."
    )
else:
    C = 0.0

# Calculate physical parameters to determine default atom scale
r_atomic, _, _ = get_crystal_properties(crystal_type, a)
default_scale_fraction = r_atomic / a

# Atom size scaling slider
atom_scale = st.sidebar.slider(
    "Atom Size Scale (Fractional)",
    0.05, 0.60, float(default_scale_fraction),
    step=0.01,
    help="Default represents the hard-sphere packing ratio."
)

# ==========================================
# 5. MAIN CONTENT - LAYOUT & UI
# ==========================================

# Main title
st.markdown("<h1 class='main-header'>⚛️ Cubic Crystal Structure & Miller Plane Visualizer</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='info-box'>"
    "Welcome! This interactive workspace is designed for Materials Science and Physics students. "
    "Select an element preset or enter custom parameters in the sidebar, input Miller indices to observe "
    "dynamic crystallographic plane intersections, and interact directly with the 3D unit cell."
    "</div>", 
    unsafe_allow_html=True
)

# Robustness & Edge Cases check
if h == 0 and k == 0 and l == 0:
    st.error("⚠️ **Input Error:** Miller Indices $(hkl)$ cannot all be zero. Please adjust the sidebar inputs.")
    st.stop()

# Calculations
d_spacing = calculate_d_spacing(a, h, k, l)
r_atomic, close_packed_dir, packing_relation = get_crystal_properties(crystal_type, a)

# Split main screen into two columns
col_left, col_right = st.columns([2, 3])

with col_left:
    st.markdown("<h2>📊 Analytical Calculations</h2>", unsafe_allow_html=True)
    
    # 3 beautiful indicator cards
    st.markdown(f"""
    <div style="display: grid; grid-template-columns: 1fr; gap: 15px;">
        <div class="calc-card">
            <div class="metric-title">Interplanar Spacing (d)</div>
            <div class="metric-value">{d_spacing:.4f} Å</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Distance between parallel planes ({h}{k}{l})</div>
        </div>
        <div class="calc-card">
            <div class="metric-title">Atomic Radius (r)</div>
            <div class="metric-value">{r_atomic:.4f} Å</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Based on hard-sphere model for {crystal_type.split()[0]}</div>
        </div>
        <div class="calc-card">
            <div class="metric-title">Close-Packed Direction</div>
            <div class="metric-value" style="color: #a855f7;">{close_packed_dir}</div>
            <div style="font-size: 0.85rem; color: #94a3b8;">Direction of maximum linear density</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Derivations and steps
    st.markdown("### 📝 Step-by-Step Derivation")
    
    with st.expander("Show Interplanar Spacing Derivation", expanded=True):
        st.write("The formula for interplanar spacing $d_{hkl}$ in a cubic crystal system is:")
        st.latex(r"d_{hkl} = \frac{a}{\sqrt{h^2 + k^2 + l^2}}")
        st.write("Substituting the current parameter values:")
        st.latex(rf"d_{{{h}{k}{l}}} = \frac{{{a:.2f}}}{{\sqrt{{({h})^2 + ({k})^2 + ({l})^2}}}}")
        h2_k2_l2 = h**2 + k**2 + l**2
        st.latex(rf"d_{{{h}{k}{l}}} = \frac{{{a:.2f}}}{{\sqrt{{{h2_k2_l2}}}}} = \frac{{{a:.2f}}}{{{np.sqrt(h2_k2_l2):.4f}}}")
        st.latex(rf"d_{{{h}{k}{l}}} \approx {d_spacing:.4f} \text{{ \AA}}")
        
    with st.expander("Show Atomic Radius & Lattice Relation", expanded=True):
        st.write(f"For a **{crystal_type}** structure, the geometric relation along the close-packed direction leads to:")
        st.latex(packing_relation)
        st.write("Substituting the current lattice parameter:")
        if crystal_type == "Simple Cubic (SC)":
            st.latex(rf"r = \frac{{{a:.2f}}}{{2}} = {r_atomic:.4f} \text{{ \AA}}")
        elif crystal_type == "Body-Centered Cubic (BCC)":
            st.latex(rf"r = \frac{{\sqrt{{3}} \times {a:.2f}}}{{4}} = \frac{{{np.sqrt(3):.4f} \times {a:.2f}}}{{4}} = {r_atomic:.4f} \text{{ \AA}}")
        else: # FCC
            st.latex(rf"r = \frac{{\sqrt{{2}} \times {a:.2f}}}{{4}} = \frac{{{np.sqrt(2):.4f} \times {a:.2f}}}{{4}} = {r_atomic:.4f} \text{{ \AA}}")

with col_right:
    st.markdown("<h2>🔮 Interactive 3D Crystal Visualization</h2>", unsafe_allow_html=True)
    st.write(f"Visualizing **{crystal_type}** lattice with Miller Plane **({h} {k} {l})** at shift factor **C = {C:.2f}**:")
    
    # ----------------------------------------------
    # Plotly 3D Figure Assembly
    # ----------------------------------------------
    fig = go.Figure()
    
    # 1. Add Cube Wireframe Edges
    edges_x = []
    edges_y = []
    edges_z = []
    for y_val in [0.0, 1.0]:
        for z_val in [0.0, 1.0]:
            edges_x.extend([0.0, 1.0, None])
            edges_y.extend([y_val, y_val, None])
            edges_z.extend([z_val, z_val, None])
    for x_val in [0.0, 1.0]:
        for z_val in [0.0, 1.0]:
            edges_x.extend([x_val, x_val, None])
            edges_y.extend([0.0, 1.0, None])
            edges_z.extend([z_val, z_val, None])
    for x_val in [0.0, 1.0]:
        for y_val in [0.0, 1.0]:
            edges_x.extend([x_val, x_val, None])
            edges_y.extend([y_val, y_val, None])
            edges_z.extend([0.0, 1.0, None])
            
    fig.add_trace(go.Scatter3d(
        x=edges_x, y=edges_y, z=edges_z,
        mode='lines',
        line=dict(color='#64748b', width=3),
        hoverinfo='none',
        name='Unit Cell Outline'
    ))
    
    # 2. Add Atoms (Lattice Points)
    atoms_coords, categories = get_lattice_points(crystal_type)
    
    # Separate corner and center/face atoms for distinct colors and descriptions
    corner_coords = atoms_coords[np.array(categories) == "Corner"]
    center_coords = atoms_coords[np.array(categories) != "Corner"]
    
    # Scaled markers based on slider
    marker_size_calc = atom_scale * 120.0
    
    # Corner atoms trace
    if len(corner_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=corner_coords[:, 0], y=corner_coords[:, 1], z=corner_coords[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size_calc,
                color='#94a3b8',
                opacity=0.9,
                line=dict(color='#0d0f13', width=2)
            ),
            text=[f"Corner Atom (Position: {p*a})" for p in corner_coords],
            hoverinfo='text',
            name='Corner Atoms (Steel Blue)'
        ))
        
    # Center/Face atoms trace
    if len(center_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=center_coords[:, 0], y=center_coords[:, 1], z=center_coords[:, 2],
            mode='markers',
            marker=dict(
                size=marker_size_calc,
                color='#f97316',
                opacity=0.9,
                line=dict(color='#0d0f13', width=2)
            ),
            text=[f"{cat} Atom (Position: {p*a})" for p, cat in zip(center_coords, [c for c in categories if c != "Corner"])],
            hoverinfo='text',
            name='Center Atoms (Coral Orange)'
        ))
        
    # 3. Add Crystallographic Plane Intersection
    intersect_pts = get_plane_intersections(h, k, l, C)
    
    if len(intersect_pts) >= 3:
        ordered_pts = order_polygon_vertices(intersect_pts, h, k, l)
        xs, ys, zs, i_idx, j_idx, k_idx = get_polygon_mesh_data(ordered_pts)
        
        # Shaded plane
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs,
            i=i_idx, j=j_idx, k=k_idx,
            color='#06b6d4',
            opacity=0.45,
            flatshading=True,
            name=f'Plane ({h}{k}{l})',
            showlegend=True
        ))
        
        # Border outline of the intersection plane
        border_x = xs + [xs[0]]
        border_y = ys + [ys[0]]
        border_z = zs + [zs[0]]
        fig.add_trace(go.Scatter3d(
            x=border_x, y=border_y, z=border_z,
            mode='lines',
            line=dict(color='#00FFFF', width=5),
            hoverinfo='none',
            name='Plane Boundary'
        ))
    else:
        st.warning(f"⚠️ **Note:** Plane $h={h}, k={k}, l={l}$ at shift $C={C:.2f}$ does not intersect the cell volume in a polygon. Try moving the 'Plane Position' slider in the sidebar.")
        
    # 4. Axes & Layout Settings
    # Labeling axis ticks with actual Angstrom measurements
    tick_vals = [0.0, 0.5, 1.0]
    tick_text = ["0", f"{a/2.0:.2f} Å", f"{a:.2f} Å"]
    
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=40),
        height=600,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(15, 23, 42, 0.8)",
            bordercolor="rgba(255, 255, 255, 0.1)",
            borderwidth=1
        ),
        scene=dict(
            xaxis=dict(
                title='X [Å]',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[-0.1, 1.1],
                gridcolor='#334155',
                zerolinecolor='#475569'
            ),
            yaxis=dict(
                title='Y [Å]',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[-0.1, 1.1],
                gridcolor='#334155',
                zerolinecolor='#475569'
            ),
            zaxis=dict(
                title='Z [Å]',
                tickvals=tick_vals,
                ticktext=tick_text,
                range=[-0.1, 1.1],
                gridcolor='#334155',
                zerolinecolor='#475569'
            ),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                up=dict(x=0, y=0, z=1)
            )
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 20px;'>"
    "Cubic Crystal Structure Visualizer & Interplanar Spacing Calculator | Developed for Materials Science Educators and Students"
    "</div>",
    unsafe_allow_html=True
)
