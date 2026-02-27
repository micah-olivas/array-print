"""
Array Print CLI

Run the full pipeline (load CSV → generate array → metrics → export .fld → plots)
from the terminal, interactively or via flags.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import typer
import questionary

from array_print import (
    generate_print_array,
    get_print_metrics,
    print_metrics_summary,
    create_project_directory,
    write_fld,
    plot_array_heatmap,
    plot_replicate_histogram,
    generate_dashboard,
)

app = typer.Typer(
    name="array-print",
    help="Generate Scienion-compatible array print files from MITOMI-assayable libraries.",
    add_completion=False,
)

PRESETS: dict[str, dict] = {
    "mitomi-ps1.8k": {"columns": 32, "rows": 56, "label": "MITOMI PS1.8k (32 × 56)"},
}


def _run_pipeline(
    project_name: str,
    library_csv: Path,
    columns: int,
    rows: int,
    skip_rows: bool,
    catalytic_binning: bool,
    plots: bool,
) -> None:
    """Execute the array-print pipeline."""
    typer.echo(f"\nProject: {project_name}")
    typer.echo(f"Library: {library_csv}")
    typer.echo(f"Array:   {columns} columns × {rows} rows")
    typer.echo(f"Options: skip_rows={skip_rows}, catalytic_binning={catalytic_binning}, plots={plots}\n")

    library_df = pd.read_csv(library_csv)

    # Build plate_well from separate Plate/Well columns if needed
    if "plate_well" not in library_df.columns and {"Plate", "Well"}.issubset(library_df.columns):
        library_df["plate_well"] = library_df["Plate"].astype(str) + library_df["Well"].astype(str)

    export_dir = create_project_directory(project_name, custom_dir=Path.cwd())

    using_blocked_device = "Block" in library_df.columns

    typer.echo("Generating print array…")
    print_array = generate_print_array(
        library_df,
        columns,
        rows,
        skip_rows,
        using_blocked_device=using_blocked_device,
        n_blocks=4,
        catalytic_binning=catalytic_binning,
    )

    metrics = get_print_metrics(print_array)
    print_metrics_summary(metrics)

    if plots:
        typer.echo("Generating plots…")
        plot_replicate_histogram(print_array, export_dir)
        plot_array_heatmap(print_array)

    typer.echo("Generating dashboard…")
    generate_dashboard(print_array, metrics, project_name, export_dir)

    print_df = pd.DataFrame(np.flip(print_array, axis=1))
    write_fld(project_name, print_df, print_df.shape[1], print_df.shape[0], export_dir)

    typer.echo(f"\nDone! Files saved to: {export_dir}")


@app.command()
def main(
    project_name: Optional[str] = typer.Argument(None, help="Name for this print project."),
    library_csv: Optional[Path] = typer.Argument(None, help="Path to library CSV file."),
    preset: str = typer.Option("mitomi-ps1.8k", help="Device preset. Use 'custom' to specify --columns/--rows."),
    columns: Optional[int] = typer.Option(None, help="Number of columns (overrides preset)."),
    rows: Optional[int] = typer.Option(None, help="Number of rows (overrides preset)."),
    skip_rows: bool = typer.Option(True, help="Insert blank rows between samples."),
    catalytic_binning: bool = typer.Option(False, help="Sort by catalytic rank before filling."),
    plots: bool = typer.Option(True, help="Generate visualization plots."),
) -> None:
    """
    Generate a Scienion-compatible .fld file from a library CSV.

    When called with no arguments, launches an interactive wizard.
    """
    # ── Interactive wizard ─────────────────────────────────────────────────────
    if project_name is None and library_csv is None:
        typer.echo("Launching interactive wizard…\n")

        project_name = questionary.text("Project name:").ask()
        if project_name is None:
            raise typer.Exit()

        raw_csv = questionary.path(
            "Library CSV path:",
            validate=lambda p: Path(p).expanduser().is_file() or "File not found",
        ).ask()
        if raw_csv is None:
            raise typer.Exit()
        library_csv = Path(raw_csv).expanduser()

        preset_choices = [p["label"] for p in PRESETS.values()] + ["Custom dimensions"]
        format_answer = questionary.select("Device format:", choices=preset_choices).ask()
        if format_answer is None:
            raise typer.Exit()

        if format_answer == "Custom dimensions":
            raw_cols = questionary.text("Number of columns:", validate=lambda v: v.isdigit() or "Enter a number").ask()
            raw_rows = questionary.text("Number of rows:", validate=lambda v: v.isdigit() or "Enter a number").ask()
            if raw_cols is None or raw_rows is None:
                raise typer.Exit()
            columns = int(raw_cols)
            rows = int(raw_rows)
        else:
            matched = next(p for p in PRESETS.values() if p["label"] == format_answer)
            columns = matched["columns"]
            rows = matched["rows"]

        option_answers = questionary.checkbox(
            "Options:",
            choices=[
                questionary.Choice("Skip rows between samples", value="skip_rows", checked=True),
                questionary.Choice("Catalytic binning (sort by rank)", value="catalytic_binning", checked=False),
                questionary.Choice("Generate visualization plots", value="plots", checked=True),
            ],
        ).ask()
        if option_answers is None:
            raise typer.Exit()

        skip_rows = "skip_rows" in option_answers
        catalytic_binning = "catalytic_binning" in option_answers
        plots = "plots" in option_answers

    # ── Non-interactive: resolve columns/rows from preset if not overridden ───
    else:
        if project_name is None or library_csv is None:
            typer.echo("Error: provide both PROJECT_NAME and LIBRARY_CSV, or neither to use the wizard.", err=True)
            raise typer.Exit(code=1)

        if not library_csv.is_file():
            typer.echo(f"Error: '{library_csv}' is not a file.", err=True)
            raise typer.Exit(code=1)

        if columns is None or rows is None:
            if preset == "custom":
                typer.echo("Error: --preset custom requires --columns and --rows.", err=True)
                raise typer.Exit(code=1)
            if preset not in PRESETS:
                typer.echo(f"Error: unknown preset '{preset}'. Available: {', '.join(PRESETS)}", err=True)
                raise typer.Exit(code=1)
            p = PRESETS[preset]
            columns = columns if columns is not None else p["columns"]
            rows = rows if rows is not None else p["rows"]

    _run_pipeline(project_name, library_csv, columns, rows, skip_rows, catalytic_binning, plots)


if __name__ == "__main__":
    app()
