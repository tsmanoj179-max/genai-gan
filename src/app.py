import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="GAN Image Synthesis Studio", layout="wide")

try:
    import torch
    from torchvision.utils import make_grid
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if not TORCH_AVAILABLE:
    st.error("### ⚠️ PyTorch is not installed in the current Python environment")
    st.info(
        "To fix this issue:\n\n"
        "1. Simply double-click **`Install_Dependencies.bat`** in the project folder to auto-install PyTorch.\n"
        "2. Or open your terminal / Anaconda Prompt and run:\n"
        "   ```bash\n"
        "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu\n"
        "   ```\n"
        "3. If using Anaconda, ensure your environment is activated:\n"
        "   ```bash\n"
        "   conda activate gan-image-project\n"
        "   ```"
    )
    st.stop()

from src.data.dataset import classifier_transforms
from src.models.classifier import MiniCNNClassifier
from src.models.gan import DCGANGenerator, denormalize_images, make_noise
from src.models.vae import ConvVAE
from src.utils.paths import ensure_dir, project_path
import os
import threading
import time


def start_auto_shutdown_monitor():
    """Shuts down the backend automatically if all browser tabs/windows are closed."""
    if getattr(start_auto_shutdown_monitor, "_started", False):
        return
    start_auto_shutdown_monitor._started = True

    def _monitor():
        # Wait until at least one browser session connects first
        has_connected = False
        while not has_connected:
            time.sleep(1)
            try:
                from streamlit.runtime import get_instance
                runtime = get_instance()
                if runtime is not None and hasattr(runtime, "_session_mgr") and runtime._session_mgr is not None:
                    if len(runtime._session_mgr.list_active_sessions()) > 0:
                        has_connected = True
            except Exception:
                pass

        # Now monitor for when the user closes the window or tab
        zero_session_count = 0
        while True:
            time.sleep(2)
            try:
                from streamlit.runtime import get_instance
                runtime = get_instance()
                if runtime is not None and hasattr(runtime, "_session_mgr") and runtime._session_mgr is not None:
                    active = runtime._session_mgr.list_active_sessions()
                    if len(active) == 0:
                        zero_session_count += 1
                        if zero_session_count >= 3:  # 6 continuous seconds of 0 connections (safe against F5 refresh)
                            os._exit(0)
                    else:
                        zero_session_count = 0
            except Exception:
                pass

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()


start_auto_shutdown_monitor()


def inject_styles(dark_mode: bool) -> None:
    if dark_mode:
        colors = {
            "page": "#030712",
            "sidebar": "#080C14",
            "surface": "rgba(17, 24, 39, 0.7)",
            "surface_soft": "rgba(31, 41, 55, 0.45)",
            "surface_high": "rgba(55, 65, 81, 0.8)",
            "text": "#F9FAFB",
            "muted": "#9CA3AF",
            "border": "rgba(255, 255, 255, 0.08)",
            "accent": "#6366F1",
            "accent_deep": "#4F46E5",
            "shadow": "rgba(99, 102, 241, 0.15)",
            "hero": "linear-gradient(135deg, rgba(17, 24, 39, 0.95) 0%, rgba(99, 102, 241, 0.08) 100%)",
            "code_bg": "#090d16",
            "code_text": "#E5E7EB",
            "ok_bg": "rgba(16, 185, 129, 0.15)",
            "ok_text": "#34D399",
            "ok_border": "rgba(16, 185, 129, 0.3)",
            "warn_bg": "rgba(245, 158, 11, 0.15)",
            "warn_text": "#FBBF24",
            "warn_border": "rgba(245, 158, 11, 0.3)",
            "arrow": "#6366F1",  # Accent color in dark mode
        }
    else:
        colors = {
            "page": "#F8FAFC",
            "sidebar": "#F1F5F9",
            "surface": "rgba(255, 255, 255, 0.75)",
            "surface_soft": "rgba(241, 245, 249, 0.6)",
            "surface_high": "rgba(226, 232, 240, 0.8)",
            "text": "#0F172A",
            "muted": "#64748B",
            "border": "rgba(0, 0, 0, 0.06)",
            "accent": "#4F46E5",
            "accent_deep": "#3730A3",
            "shadow": "rgba(79, 70, 229, 0.05)",
            "hero": "linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(79, 70, 229, 0.05) 100%)",
            "code_bg": "#1E293B",
            "code_text": "#F8FAFC",
            "ok_bg": "rgba(16, 185, 129, 0.08)",
            "ok_text": "#059669",
            "ok_border": "rgba(16, 185, 129, 0.2)",
            "warn_bg": "rgba(245, 158, 11, 0.08)",
            "warn_text": "#D97706",
            "warn_border": "rgba(245, 158, 11, 0.2)",
            "arrow": "#0F172A",  # Text main (normal) color in light mode
        }

    st.markdown(
        f"""
        <style>
        :root {{
            --page-bg: {colors["page"]};
            --sidebar-bg: {colors["sidebar"]};
            --surface: {colors["surface"]};
            --surface-soft: {colors["surface_soft"]};
            --surface-high: {colors["surface_high"]};
            --text-main: {colors["text"]};
            --text-muted: {colors["muted"]};
            --line: {colors["border"]};
            --accent: {colors["accent"]};
            --accent-deep: {colors["accent_deep"]};
            --shadow: {colors["shadow"]};
            --hero-bg: {colors["hero"]};
            --code-bg: {colors["code_bg"]};
            --code-text: {colors["code_text"]};
            --ok-bg: {colors["ok_bg"]};
            --ok-text: {colors["ok_text"]};
            --ok-border: {colors["ok_border"]};
            --warn-bg: {colors["warn_bg"]};
            --warn-text: {colors["warn_text"]};
            --warn-border: {colors["warn_border"]};
            --arrow-color: {colors["arrow"]};
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800;900&family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

        /* Typography & Globals */
        html, body, [class*="css"], .stApp {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--page-bg);
            color: var(--text-main);
        }

        h1, h2, h3, h4, h5, h6, .hero-title, .section-label {
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
            font-weight: 700 !important;
        }

        /* Targeted Global transitions for Theme Switch toggling */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"],
        [data-testid="stSidebar"] > div:first-child,
        .hero-panel,
        .soft-panel,
        .spec-item,
        .status-pill,
        .note-box,
        .result-box,
        .command-box,
        .stButton > button,
        button[data-baseweb="tab"],
        div[data-testid="stFileUploader"] {
            transition: background 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), 
                        background-color 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), 
                        border-color 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), 
                        box-shadow 0.4s cubic-bezier(0.25, 0.8, 0.25, 1), 
                        transform 0.3s cubic-bezier(0.25, 0.8, 0.25, 1),
                        color 0.4s cubic-bezier(0.25, 0.8, 0.25, 1);
        }

        [data-testid="stAppViewContainer"],
        [data-testid="stHeader"] {
            background: var(--page-bg);
            color: var(--text-main);
        }

        [data-testid="stHeader"] {
            border-bottom: 1px solid var(--line);
            backdrop-filter: blur(12px);
        }

        /* Solid background for Sidebar to avoid visual bleeding */
        [data-testid="stSidebar"] {
            background-color: var(--sidebar-bg) !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            background-color: var(--sidebar-bg) !important;
            border-right: 1px solid var(--line);
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        h1, h2, h3, h4, h5, h6, p, label,
        [data-testid="stMarkdownContainer"],
        [data-testid="stWidgetLabel"],
        [data-testid="stCaptionContainer"] {
            color: var(--text-main);
        }

        [data-testid="stCaptionContainer"],
        [data-testid="stWidgetLabel"] p {
            color: var(--text-muted);
            font-size: 0.85rem;
        }

        /* Metrics */
        div[data-testid="stMetric"] {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1rem 1.25rem;
            box-shadow: 0 10px 25px var(--shadow);
            backdrop-filter: blur(8px);
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            border-color: var(--accent);
            box-shadow: 0 15px 35px var(--shadow), 0 0 0 1px rgba(99, 102, 241, 0.1);
        }

        div[data-testid="stMetricLabel"] {
            color: var(--text-muted);
            font-size: 0.82rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-main);
            font-size: 1.45rem;
            font-weight: 700;
            margin-top: 0.25rem;
        }

        /* Custom Panel elements */
        .hero-panel {
            background: var(--hero-bg);
            border: 1px solid var(--line);
            border-left: 6px solid var(--accent);
            border-radius: 16px;
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            backdrop-filter: blur(12px);
            box-shadow: 0 12px 30px var(--shadow);
        }

        .hero-row {
            display: flex;
            justify-content: space-between;
            gap: 1.5rem;
            align-items: center;
            margin-bottom: 0.5rem;
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }

        .theme-badge {
            background: var(--surface-high);
            border: 1px solid var(--line);
            border-radius: 99px;
            color: var(--text-main);
            display: inline-block;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 0.35rem 0.8rem;
            white-space: nowrap;
            letter-spacing: 0.02em;
        }

        .hero-title {
            color: var(--text-main);
            font-size: 2.15rem;
            margin: 0;
            line-height: 1.2;
            letter-spacing: -0.02em;
        }

        .hero-copy {
            color: var(--text-muted);
            font-size: 1.05rem;
            line-height: 1.6;
            margin: 0;
            max-width: 900px;
        }

        .section-label {
            color: var(--text-main);
            font-size: 1.15rem;
            font-weight: 700;
            margin: 0.5rem 0 0.75rem 0;
            letter-spacing: -0.01em;
        }

        .soft-panel {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 16px;
            padding: 1.25rem;
            min-height: 100%;
            backdrop-filter: blur(12px);
            box-shadow: 0 10px 30px var(--shadow);
        }

        .soft-panel:hover {
            transform: translateY(-4px);
            box-shadow: 0 16px 40px var(--shadow), 0 0 0 1px rgba(99, 102, 241, 0.1);
            border-color: rgba(99, 102, 241, 0.3);
        }

        .spec-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin-top: 0.5rem;
        }

        .spec-item {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 0.85rem 1rem;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }

        .spec-item:hover {
            background: var(--surface-high);
            transform: translateY(-2px);
            box-shadow: 0 8px 20px var(--shadow);
            border-color: var(--accent);
        }

        .spec-label {
            color: var(--text-muted);
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 0.25rem;
        }

        .spec-value {
            color: var(--text-main);
            font-size: 0.95rem;
            font-weight: 700;
            line-height: 1.4;
        }

        .status-pill {
            display: inline-flex;
            align-items: center;
            border-radius: 99px;
            padding: 0.25rem 0.75rem;
            font-size: 0.74rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            border: 1px solid var(--line);
            background: var(--surface-soft);
            color: var(--text-muted);
        }

        .status-ok {
            background: var(--ok-bg);
            color: var(--ok-text);
            border-color: var(--ok-border);
            box-shadow: 0 0 12px rgba(16, 185, 129, 0.1);
        }

        .status-missing {
            background: var(--warn-bg);
            color: var(--warn-text);
            border-color: var(--warn-border);
            box-shadow: 0 0 12px rgba(245, 158, 11, 0.1);
        }

        .note-box {
            background: var(--surface-soft);
            border: 1px solid var(--line);
            border-radius: 12px;
            color: var(--text-muted);
            font-size: 0.92rem;
            line-height: 1.55;
            padding: 1rem 1.15rem;
            margin-top: 0.75rem;
        }

        .result-box {
            background: var(--surface);
            border: 1px solid var(--line);
            border-radius: 12px;
            color: var(--text-main);
            padding: 1.15rem;
            margin-top: 0.75rem;
            box-shadow: 0 8px 24px var(--shadow);
        }

        .path-text {
            color: var(--text-muted);
            font-family: Consolas, Monaco, monospace;
            font-size: 0.8rem;
            overflow-wrap: anywhere;
        }

        .command-box {
            background: var(--code-bg);
            color: var(--code-text);
            border-radius: 12px;
            padding: 1rem 1.25rem;
            font-family: Consolas, Monaco, monospace;
            font-size: 0.84rem;
            line-height: 1.55;
            overflow-wrap: anywhere;
            border: 1px solid var(--line);
        }

        /* File Uploader styling */
        div[data-testid="stFileUploader"] {
            background: var(--surface);
            border: 1px dashed var(--accent);
            border-radius: 16px;
            padding: 0.75rem;
            box-shadow: 0 6px 20px var(--shadow);
        }

        div[data-testid="stFileUploader"] section {
            background: var(--surface-soft);
            border: none !important;
            border-radius: 12px;
            color: var(--text-main);
            padding: 1.5rem;
        }

        /* File Uploader text color corrections in dark & light themes */
        div[data-testid="stFileUploader"] * {
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] span, 
        div[data-testid="stFileUploader"] small, 
        div[data-testid="stFileUploader"] p {
            color: var(--text-muted) !important;
        }
        div[data-testid="stFileUploader"] h4,
        div[data-testid="stFileUploader"] strong {
            color: var(--text-main) !important;
        }
        div[data-testid="stFileUploader"] button {
            background-color: var(--surface-soft) !important;
            border: 1px solid var(--line) !important;
            color: var(--text-main) !important;
            border-radius: 8px !important;
            padding: 0.4rem 0.8rem !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            background-color: var(--surface-high) !important;
            color: var(--accent) !important;
        }

        /* Buttons Styling */
        .stButton > button {
            border-radius: 10px !important;
            min-height: 2.85rem;
            font-weight: 700 !important;
            font-family: 'Outfit', sans-serif !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-size: 0.85rem !important;
            background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%) !important;
            color: white !important;
            border: none !important;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
            cursor: pointer;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 22px rgba(99, 102, 241, 0.45) !important;
        }

        .stButton > button:active {
            transform: translateY(0) !important;
            box-shadow: 0 2px 10px rgba(99, 102, 241, 0.2) !important;
        }

        /* Tab bar styling (iOS Segmented style) */
        div[data-baseweb="tab-list"] {
            background: var(--surface-soft);
            border-radius: 12px;
            padding: 0.35rem;
            gap: 0.5rem;
            border: 1px solid var(--line);
            margin-bottom: 1.5rem;
        }

        button[data-baseweb="tab"] {
            background: transparent !important;
            border: none !important;
            border-radius: 8px !important;
            color: var(--text-muted) !important;
            padding: 0.6rem 1.25rem !important;
            font-weight: 600 !important;
            font-size: 0.9rem !important;
        }

        button[data-baseweb="tab"]:hover {
            color: var(--text-main) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            background: var(--surface) !important;
            color: var(--accent) !important;
            box-shadow: 0 4px 12px var(--shadow) !important;
            font-weight: 700 !important;
        }

        div[data-baseweb="tab-highlight-line"] {
            background-color: transparent !important; /* Hide default base-web line */
        }

        /* Global Form elements readability & contrast fixes */
        input, textarea, select {
            color: var(--text-main) !important;
        }

        hr {
            border-color: var(--line) !important;
            opacity: 0.4;
        }

        /* Radio Buttons */
        [data-baseweb="radio"] label {
            color: var(--text-main) !important;
            font-weight: 500;
        }
        [data-baseweb="radio"] div[role="presentation"] {
            border-color: var(--line) !important;
            background-color: var(--surface-soft) !important;
        }
        [data-baseweb="radio"] div[aria-checked="true"] div[role="presentation"] {
            border-color: var(--accent) !important;
            background-color: var(--accent) !important;
        }

        /* Toggle Switches */
        div[data-testid="stToggle"] label p {
            color: var(--text-main) !important;
            font-weight: 600;
        }
        div[data-testid="stToggle"] div[role="switch"] {
            background-color: var(--surface-soft) !important;
            border: 1px solid var(--line) !important;
        }
        div[data-testid="stToggle"] div[role="switch"][aria-checked="true"] {
            background-color: var(--accent) !important;
        }
        div[data-testid="stToggle"] div[role="switch"] div {
            background-color: var(--text-main) !important;
        }

        /* Selectboxes */
        div[data-testid="stSelectbox"] label p {
            color: var(--text-main) !important;
            font-weight: 600;
        }
        div[data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background-color: var(--surface-soft) !important;
            border: 1px solid var(--line) !important;
            color: var(--text-main) !important;
            border-radius: 10px !important;
        }

        /* Number Inputs */
        div[data-testid="stNumberInput"] label p {
            color: var(--text-main) !important;
            font-weight: 600;
        }
        div[data-testid="stNumberInput"] [data-baseweb="input"] > div {
            background-color: var(--surface-soft) !important;
            border: 1px solid var(--line) !important;
            color: var(--text-main) !important;
            border-radius: 10px !important;
        }
        div[data-testid="stNumberInput"] button {
            background-color: var(--surface-soft) !important;
            color: var(--text-main) !important;
            border: 1px solid var(--line) !important;
        }
        div[data-testid="stNumberInput"] button:hover {
            background-color: var(--surface-high) !important;
            color: var(--accent) !important;
        }

        /* Dropdown Popovers & Streamlit native 3-dot dropdown menu */
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] > div,
        div[data-baseweb="menu"] ul,
        div[data-baseweb="menu"] li,
        div[role="listbox"],
        div[role="listbox"] > div,
        div[role="listbox"] ul,
        div[role="listbox"] li,
        ul[role="listbox"],
        ul[role="listbox"] li,
        div[role="menu"],
        div[role="menu"] > div,
        div[role="menu"] li,
        div[role="option"] {
            background-color: var(--surface-high) !important;
            color: var(--text-main) !important;
            border-color: var(--line) !important;
        }

        /* Specific container borders & roundings */
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        div[role="menu"],
        ul[role="listbox"],
        div[role="listbox"] {
            border: 1px solid var(--line) !important;
            border-radius: 10px !important;
            box-shadow: 0 12px 36px var(--shadow) !important;
        }

        /* Transparent background by default for lists & buttons inside portal menus */
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"] li,
        div[role="menuitem"],
        ul[role="listbox"] li,
        div[role="option"],
        div[data-baseweb="popover"] button {
            background-color: transparent !important;
            color: var(--text-main) !important;
            padding: 0.6rem 1rem !important;
            border: none !important;
        }

        /* Ensure child tags inside option tags also color correctly */
        div[data-baseweb="popover"] li *,
        div[data-baseweb="menu"] li *,
        div[role="menuitem"] *,
        ul[role="listbox"] li *,
        div[role="option"] *,
        div[data-baseweb="popover"] button * {
            color: var(--text-main) !important;
            background-color: transparent !important;
        }

        /* Hover states for lists & buttons in portal menus */
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="popover"] li:hover *,
        div[data-baseweb="menu"] li:hover,
        div[data-baseweb="menu"] li:hover *,
        div[role="menuitem"]:hover,
        div[role="menuitem"]:hover *,
        ul[role="listbox"] li:hover,
        ul[role="listbox"] li:hover *,
        div[role="option"]:hover,
        div[role="option"]:hover *,
        div[data-baseweb="popover"] button:hover,
        div[data-baseweb="popover"] button:hover * {
            background-color: var(--surface-soft) !important;
            color: var(--accent) !important;
        }

        button[data-testid="stHeaderMainMenu"] svg * {
            fill: var(--text-main) !important;
            color: var(--text-main) !important;
        }

        /* Sidebar Collapse/Expand control arrows & any chevron arrows visibility */
        button[data-testid="stSidebarCollapseButton"] svg,
        button[data-testid="stSidebarCollapseButton"] svg *,
        div[data-testid="collapsedControl"] svg,
        div[data-testid="collapsedControl"] svg *,
        div[data-testid="collapsedControl"] button,
        button[data-testid="stSidebarCollapseButton"] {
            fill: var(--text-main) !important;
            stroke: var(--text-main) !important;
            color: var(--text-main) !important;
        }
        button[data-testid="stSidebarCollapseButton"]:hover,
        div[data-testid="collapsedControl"] button:hover {
            color: var(--accent) !important;
            background-color: var(--surface-soft) !important;
        }

        /* Selectbox dropdown arrow icon visibility */
        div[data-testid="stSelectbox"] svg,
        div[data-testid="stSelectbox"] svg * {
            fill: var(--text-main) !important;
            stroke: var(--text-main) !important;
        }

        /* Sliders */
        div[data-testid="stSlider"] label p {
            color: var(--text-main) !important;
            font-weight: 600;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] {
            margin-top: 0.5rem;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child {
            background-color: var(--surface-soft) !important;
            height: 6px !important;
            border-radius: 99px;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div:first-child > div:first-child {
            background-color: var(--accent) !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: var(--accent) !important;
            border: 2px solid var(--text-main) !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3) !important;
            width: 20px !important;
            height: 20px !important;
        }
        div[data-testid="stSlider"] div[role="slider"] ~ div {
            color: var(--text-main) !important;
            font-weight: 700 !important;
        }

        /* Images and media elements */
        div[data-testid="stImage"] img {
            border-radius: 12px;
            border: 1px solid var(--line);
            box-shadow: 0 8px 24px var(--shadow);
        }

        /* Progress Bar styling */
        div[data-testid="stProgress"] > div > div > div > div {
            background: linear-gradient(90deg, var(--accent) 0%, var(--accent-deep) 100%) !important;
            border-radius: 99px;
        }
        div[data-testid="stProgress"] {
            height: 8px !important;
        }

        /* Sidebar Subheader and caption colors */
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p {
            color: var(--text-main) !important;
        }
        [data-testid="stSidebar"] span[data-testid="stCaptionContainer"] {
            color: var(--text-muted) !important;
        }

        @media (max-width: 760px) {
            .spec-grid {
                grid-template-columns: 1fr;
            }
            .hero-panel {
                padding: 1.25rem;
            }
            .hero-title {
                font-size: 1.65rem;
            }
            .hero-row {
                flex-direction: column;
                align-items: flex-start;
                gap: 0.75rem;
            }
            .theme-badge {
                margin-top: 0.25rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



def render_status(label: str, path: Path) -> None:
    available = path.exists()
    css_class = "status-ok" if available else "status-missing"
    text = "Ready" if available else "Not trained"
    st.markdown(
        f"""
        <div class="soft-panel">
            <div class="spec-label">{label}</div>
            <div class="spec-value">{path.name}</div>
            <div style="margin-top:0.55rem;">
                <span class="status-pill {css_class}">{text}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
def render_spec_grid() -> None:
    st.markdown(
        """
        <div class="soft-panel">
            <div class="section-label">Recommended Project Specs</div>
            <div class="spec-grid">
                <div class="spec-item">
                    <div class="spec-label">System target</div>
                    <div class="spec-value">Intel i7-13620H, 16 GB RAM, CPU-only</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Dataset</div>
                    <div class="spec-value">30k real + 30k fake images</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Best practical GAN</div>
                    <div class="spec-value">Compact DCGAN at 128 x 128</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Balanced GAN run</div>
                    <div class="spec-value">20 epochs, batch size 16</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">Classifier training</div>
                    <div class="spec-value">Uses both real and fake images</div>
                </div>
                <div class="spec-item">
                    <div class="spec-label">GAN and VAE training</div>
                    <div class="spec-value">Uses real images only</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def tensor_grid_to_pil(images: torch.Tensor, nrow: int = 4) -> Image.Image:
    grid = make_grid(images, nrow=nrow, padding=2)
    array = grid.permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip(array * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(array)


def checkpoint_status(path: Path) -> str:
    return "Ready" if path.exists() else "Not trained"


def load_classifier(checkpoint_path: Path):
    if not checkpoint_path.exists():
        return None, None
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model = MiniCNNClassifier()
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def load_gan(checkpoint_path: Path, fallback_image_size: int, fallback_latent_dim: int, fallback_feature_maps: int):
    trained = checkpoint_path.exists()
    if trained:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        image_size = int(config.get("image_size", fallback_image_size))
        latent_dim = int(config.get("latent_dim", fallback_latent_dim))
        feature_maps = int(config.get("feature_maps", fallback_feature_maps))
    else:
        image_size = fallback_image_size
        latent_dim = fallback_latent_dim
        feature_maps = fallback_feature_maps
    model = DCGANGenerator(latent_dim=latent_dim, feature_maps=feature_maps, image_size=image_size)
    if trained:
        model.load_state_dict(checkpoint["generator"])
    model.eval()
    return model, {"image_size": image_size, "latent_dim": latent_dim, "feature_maps": feature_maps}, trained


def load_vae(checkpoint_path: Path, fallback_image_size: int, fallback_latent_dim: int, fallback_base_channels: int):
    trained = checkpoint_path.exists()
    if trained:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        image_size = int(config.get("image_size", fallback_image_size))
        latent_dim = int(config.get("latent_dim", fallback_latent_dim))
        base_channels = int(config.get("base_channels", fallback_base_channels))
    else:
        image_size = fallback_image_size
        latent_dim = fallback_latent_dim
        base_channels = fallback_base_channels
    model = ConvVAE(latent_dim=latent_dim, base_channels=base_channels, image_size=image_size)
    if trained:
        model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, {"image_size": image_size, "latent_dim": latent_dim, "base_channels": base_channels}, trained


def save_generated_grid(image: Image.Image, method: str) -> Path:
    output_dir = ensure_dir(project_path("outputs", "samples"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"app_{method}_{timestamp}.png"
    image.save(output_path)
    return output_path


with st.sidebar:
    st.subheader("Appearance")
    dark_mode = st.toggle("Dark Mode", value=False, key="theme_toggle")

theme_name = "Dark" if dark_mode else "Light"

inject_styles(dark_mode)

st.markdown(
    f"""
    <div class="hero-panel">
        <div class="hero-row">
            <div>
                <div class="hero-kicker">CPU-ready synthesis studio</div>
                <div class="hero-title">Generative AI for Image Synthesis Using GANs</div>
            </div>
            <div class="theme-badge">{theme_name} theme</div>
        </div>
        <p class="hero-copy">
            A CPU-friendly image synthesis and detection studio: generate images with a trained DCGAN or VAE,
            then classify uploaded images as real or AI-generated using your trained real/fake classifier.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

gan_checkpoint = project_path("models", "gan", "dcgan_latest.pt")
vae_checkpoint = project_path("models", "vae", "vae_latest.pt")
classifier_checkpoint = project_path("models", "classifier", "classifier_best.pt")

with st.sidebar:
    st.subheader("Run Settings")
    st.caption("Used only when a checkpoint is not available.")
    fallback_image_size = st.selectbox("Image size", [64, 128, 256], index=1)
    fallback_latent_dim = st.slider("Latent size", 32, 256, 128, step=32)
    st.divider()
    st.subheader("Checkpoint Status")
    st.caption(f"GAN: {checkpoint_status(gan_checkpoint)}")
    st.caption(f"VAE: {checkpoint_status(vae_checkpoint)}")
    st.caption(f"Classifier: {checkpoint_status(classifier_checkpoint)}")

status_a, status_b, status_c = st.columns(3)
with status_a:
    render_status("GAN generator", gan_checkpoint)
with status_b:
    render_status("VAE generator", vae_checkpoint)
with status_c:
    render_status("Real/fake classifier", classifier_checkpoint)

st.write("")
render_spec_grid()
st.write("")

generate_tab, classify_tab, workflow_tab = st.tabs(["Generate Image", "Classify Image", "Training Workflow"])

with generate_tab:
    left, right = st.columns([0.95, 2.05], gap="large")
    with left:
        st.markdown('<div class="section-label">Generation Controls</div>', unsafe_allow_html=True)
        method = st.radio("Method", ["GAN", "VAE"], horizontal=True)
        count = st.slider("Images", 1, 16, 4)
        generate = st.button("Generate Images", type="primary", use_container_width=True)
        st.markdown(
            """
            <div class="note-box">
                Use GAN for the final project output. VAE is included as a second synthesis method and usually
                produces softer images than the GAN.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown('<div class="section-label">Generated Output</div>', unsafe_allow_html=True)
        if generate:
            with st.spinner("Generating..."):
                if method == "GAN":
                    model, config, trained = load_gan(
                        gan_checkpoint,
                        fallback_image_size=fallback_image_size,
                        fallback_latent_dim=fallback_latent_dim,
                        fallback_feature_maps=32,
                    )
                    noise = make_noise(count, config["latent_dim"], torch.device("cpu"))
                    with torch.no_grad():
                        images = denormalize_images(model(noise))
                    grid_image = tensor_grid_to_pil(images, nrow=min(4, count))
                    saved_path = save_generated_grid(grid_image, "gan")
                else:
                    model, config, trained = load_vae(
                        vae_checkpoint,
                        fallback_image_size=fallback_image_size,
                        fallback_latent_dim=fallback_latent_dim,
                        fallback_base_channels=32,
                    )
                    with torch.no_grad():
                        images = model.sample(count, torch.device("cpu"))
                    grid_image = tensor_grid_to_pil(images, nrow=min(4, count))
                    saved_path = save_generated_grid(grid_image, "vae")

            if not trained:
                st.info("No trained checkpoint found yet. The image grid came from an untrained model.")
            st.image(grid_image, use_container_width=True)
            st.markdown(
                f"""
                <div class="result-box">
                    <strong>Saved output</strong><br>
                    <span class="path-text">{saved_path}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="note-box">
                    Choose a method and click Generate Images. If a trained checkpoint exists, this panel will
                    show images from your trained model.
                </div>
                """,
                unsafe_allow_html=True,
            )

with classify_tab:
    classifier, classifier_config = load_classifier(classifier_checkpoint)
    left, right = st.columns([1, 1], gap="large")
    with left:
        st.markdown('<div class="section-label">Upload Image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Image", type=["jpg", "jpeg", "png", "bmp", "webp"], label_visibility="collapsed")

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        with left:
            st.image(image, use_container_width=True)

        with right:
            st.markdown('<div class="section-label">Classification Result</div>', unsafe_allow_html=True)
            if classifier is None:
                st.info("Classifier checkpoint is not available yet. Train it with python -m src.train_classifier.")
            else:
                image_size = int(classifier_config.get("image_size", fallback_image_size))
                transform = classifier_transforms(image_size=image_size, train=False)
                tensor = transform(image).unsqueeze(0)
                with torch.no_grad():
                    logit = classifier(tensor)
                    prob_fake = torch.sigmoid(logit).item()
                prob_real = 1.0 - prob_fake
                label = "Fake / AI-generated" if prob_fake >= 0.5 else "Real"

                metric_a, metric_b = st.columns(2)
                with metric_a:
                    st.metric("Prediction", label)
                with metric_b:
                    st.metric("Fake confidence", f"{prob_fake:.2%}")
                st.progress(float(prob_fake), text="Fake probability")
                st.markdown(
                    f"""
                    <div class="result-box">
                        <strong>Probability breakdown</strong><br>
                        Real image: {prob_real:.2%}<br>
                        Fake or AI-generated image: {prob_fake:.2%}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    else:
        with right:
            st.markdown('<div class="section-label">Classification Result</div>', unsafe_allow_html=True)
            st.markdown(
                """
                <div class="note-box">
                    Upload a JPG, PNG, BMP, or WebP image. The trained classifier will show real and fake
                    probabilities here.
                </div>
                """,
                unsafe_allow_html=True,
            )

with workflow_tab:
    st.markdown('<div class="section-label">Recommended Training Order</div>', unsafe_allow_html=True)
    order_a, order_b, order_c, order_d = st.columns(4)
    with order_a:
        st.metric("1", "Preprocess")
    with order_b:
        st.metric("2", "Classifier")
    with order_c:
        st.metric("3", "GAN")
    with order_d:
        st.metric("4", "VAE optional")

    cmd_left, cmd_right = st.columns(2, gap="large")
    with cmd_left:
        st.markdown('<div class="section-label">Balanced CPU Commands</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="command-box">
            python -m src.data.prepare_dataset --real-dir data/raw/real --fake-dir data/raw/fake --image-size 128<br><br>
            python -m src.train_classifier --epochs 8 --batch-size 32 --image-size 128<br><br>
            python -m src.train_gan --epochs 40 --batch-size 16 --image-size 128
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cmd_right:
        st.markdown('<div class="section-label">Expected Practical Result</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="note-box">
                The classifier uses the full real/fake dataset. The GAN uses real images only, because its task is
                to learn the real image distribution and generate new face-like samples. At 128 x 128, 40 epochs
                is the balanced target for your CPU-only laptop.
            </div>
            """,
            unsafe_allow_html=True,
        )
