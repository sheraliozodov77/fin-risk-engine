"""
EDA utilities: schema audit, summary statistics, and professional plot styling.
"""
from .schema_audit import schema_audit_df, data_quality_checks
from .summaries import numeric_summary, categorical_summary
from .plot_style import apply_style, get_colors, PALETTE, CATEGORY_PALETTE

__all__ = [
    "schema_audit_df",
    "data_quality_checks",
    "numeric_summary",
    "categorical_summary",
    "apply_style",
    "get_colors",
    "PALETTE",
    "CATEGORY_PALETTE",
]
