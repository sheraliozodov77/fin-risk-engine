"""
Professional plot styling for EDA and reports.
Fintech/corporate palette; consistent typography and grid.
"""
from __future__ import annotations

# Fintech/corporate palette: primary, secondary, accent (alert), neutral
PALETTE = {
    "primary": "#1e3a5f",      # navy
    "secondary": "#2d6a6e",    # teal
    "accent": "#c75c41",       # coral (alerts, fraud)
    "neutral": "#6b7280",      # gray
    "light": "#e5e7eb",
    "white": "#ffffff",
}
COLORS = list(PALETTE.values())

# Categorical palette (max 6–8 distinct)
CATEGORY_PALETTE = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["neutral"], "#4a7c59", "#7c5a9e"]


def apply_style(
    style: str = "whitegrid",
    context: str = "notebook",
    font_scale: float = 1.15,
    figsize_default: tuple[float, float] = (10, 5),
    dpi: int = 120,
) -> None:
    """
    Apply professional matplotlib + seaborn style.
    Call once at notebook start.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_theme(
        style=style,
        context=context,
        font_scale=font_scale,
        rc={
            "figure.figsize": figsize_default,
            "figure.dpi": dpi,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.titlesize": 14,
            "axes.titleweight": "600",
            "axes.labelweight": "500",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "font.family": ["sans-serif"],
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        },
    )
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=CATEGORY_PALETTE)
    plt.rcParams["figure.facecolor"] = PALETTE["white"]
    plt.rcParams["axes.facecolor"] = PALETTE["white"]


def get_colors():
    """Return palette dict and list for use in plots."""
    return PALETTE, COLORS, CATEGORY_PALETTE
