"""
Array Print CLI

Run the full pipeline (load CSV → generate array → metrics → export .fld → plots)
from the terminal, interactively or via flags.

Also supports converting .fld files back to pinlists.
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
    plot_library_panels,
    plot_array_map,
    parse_fld,
    fld_to_pinlist,
)

app = typer.Typer(
    name="array-print",
    help="Generate Scienion-compatible array print files from MITOMI-assayable libraries.",
    add_completion=False,
    invoke_without_command=True,
)

PRESETS: dict[str, dict] = {
    "mitomi-ps1.8k": {
        "columns": 32, "rows": 56,
        "label": "MITOMI PS1.8k (32 × 56)",
        "device": "MITOMI PS1.8k",
        "fld_device": "PS1.8K",
    },
}


def _run_pipeline(
    project_name: str,
    library_csv: Path,
    columns: int,
    rows: int,
    skip_rows: bool,
    catalytic_binning: bool,
    plots: bool,
    category_column: Optional[str] = None,
    group_column: Optional[str] = None,
    refresh: int = 0,
    device: Optional[str] = None,
    fld_device: Optional[str] = None,
    seed: int = 0,
    write_pinlist: bool = True,
) -> None:
    """Execute the array-print pipeline."""
    typer.echo(f"\nProject: {project_name}")
    typer.echo(f"Library: {library_csv}")
    typer.echo(f"Array:   {columns} columns × {rows} rows")
    typer.echo(f"Options: skip_rows={skip_rows}, catalytic_binning={catalytic_binning}, plots={plots}")
    if category_column:
        typer.echo(f"Colour by: {category_column}")
    if group_column:
        typer.echo(f"Grouped by: {group_column}")
    typer.echo("")

    library_df = pd.read_csv(library_csv)

    # Build plate_well from separate Plate/Well columns if needed
    if "plate_well" not in library_df.columns and {"Plate", "Well"}.issubset(library_df.columns):
        library_df["plate_well"] = library_df["Plate"].astype(str) + library_df["Well"].astype(str)

    # Variant → category, used to give each group its own colour on the map
    categories = None
    if category_column:
        if category_column not in library_df.columns:
            available = ", ".join(library_df.columns)
            typer.echo(
                f"Error: no column named '{category_column}' in {library_csv.name}. "
                f"Columns present: {available}",
                err=True,
            )
            raise typer.Exit(code=1)
        if "Name" not in library_df.columns:
            typer.echo(
                "Error: --category-column needs a 'Name' column to map categories onto variants.",
                err=True,
            )
            raise typer.Exit(code=1)
        categories = (
            library_df.dropna(subset=["Name"])
            .set_index("Name")[category_column]
            .to_dict()
        )

    # Second designation: which kind of sublibrary each category belongs to.
    category_groups = None
    if group_column:
        if not category_column:
            typer.echo("Error: --group-column needs --category-column as well.", err=True)
            raise typer.Exit(code=1)
        if group_column not in library_df.columns:
            available = ", ".join(library_df.columns)
            typer.echo(
                f"Error: no column named '{group_column}' in {library_csv.name}. "
                f"Columns present: {available}",
                err=True,
            )
            raise typer.Exit(code=1)
        category_groups = (
            library_df.dropna(subset=[category_column, group_column])
            .groupby(category_column)[group_column]
            .first()
            .astype(str)
            .to_dict()
        )

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
        seed=seed,
    )

    metrics = get_print_metrics(print_array)
    print_metrics_summary(metrics)

    # Map well IDs → variant names for all visualizations
    if 'plate_well' in library_df.columns and 'Name' in library_df.columns:
        well_to_name = library_df.set_index('plate_well')['Name'].to_dict()
        display_array = np.vectorize(
            lambda w: well_to_name.get(w, w) if w != '' else ''
        )(print_array)
    else:
        display_array = print_array

    if plots:
        typer.echo("Generating plots…")
        plot_replicate_histogram(display_array, export_dir)
        plot_array_heatmap(display_array)
        plot_array_map(
            display_array,
            categories=categories,
            n_blocks=4 if using_blocked_device else None,
            category_groups=category_groups,
            export_dir=export_dir,
            project_name=project_name,
        )
        if categories:
            plot_library_panels(
                display_array,
                categories=categories,
                n_blocks=4 if using_blocked_device else None,
                category_groups=category_groups,
                export_dir=export_dir,
                project_name=project_name,
            )

    typer.echo("Generating dashboard…")
    generate_dashboard(
        display_array, metrics, project_name, export_dir,
        categories=categories,
        n_blocks=4 if using_blocked_device else None,
        category_groups=category_groups,
        refresh=refresh,
        device=device,
    )

    print_df = pd.DataFrame(np.flip(print_array, axis=1))
    fld_path = write_fld(project_name, print_df, print_df.shape[1], print_df.shape[0],
                         export_dir, device=fld_device)

    if write_pinlist:
        pinlist_path = Path(export_dir) / f"{project_name}_pinlist.csv"
        fld_to_pinlist(fld_path=fld_path, library_csv=str(library_csv),
                       output=str(pinlist_path))
        typer.echo(f"Pinlist written: {pinlist_path.name}")

    typer.echo(f"\nDone! Files saved to: {export_dir}")


@app.callback()
def callback(ctx: typer.Context) -> None:
    """
    Array Print: generate and convert Scienion-compatible print files.

    Run with no subcommand to launch the generate wizard.
    """
    if ctx.invoked_subcommand is None:
        # Launch the generate wizard directly
        _interactive_generate()


def _interactive_generate() -> None:
    """Run the generate wizard interactively."""
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
        device = fld_device = None
    else:
        matched = next(p for p in PRESETS.values() if p["label"] == format_answer)
        columns = matched["columns"]
        rows = matched["rows"]
        device = matched.get("device")
        fld_device = matched.get("fld_device")

    option_answers = questionary.checkbox(
        "Options:",
        choices=[
            questionary.Choice("Skip rows between samples", value="skip_rows", checked=True),
            questionary.Choice("Catalytic binning (sort by rank)", value="catalytic_binning", checked=False),
            questionary.Choice("Generate visualization plots", value="plots", checked=True),
            questionary.Choice("Write pinlist", value="pinlist", checked=True),
        ],
    ).ask()
    if option_answers is None:
        raise typer.Exit()

    skip_rows = "skip_rows" in option_answers
    catalytic_binning = "catalytic_binning" in option_answers
    plots = "plots" in option_answers
    write_pinlist = "pinlist" in option_answers

    category_column = _ask_category_column(library_csv)
    group_column = (
        _ask_category_column(library_csv, category_column) if category_column else None
    )

    _run_pipeline(
        project_name, library_csv, columns, rows, skip_rows, catalytic_binning, plots,
        category_column=category_column, group_column=group_column, device=device,
        fld_device=fld_device, write_pinlist=write_pinlist,
    )


def _ask_category_column(library_csv: Path,
                         exclude: Optional[str] = None) -> Optional[str]:
    """
    Offer the library's grouping columns.

    Called once for the colour category, then again for the optional second
    designation that sub-heads the legend's sublibraries.
    """
    try:
        header = pd.read_csv(library_csv, nrows=200)
    except Exception:
        return None

    skip = {"Name", "plate_well", "Plate", "Well"} | ({exclude} if exclude else set())
    # A column worth colouring by repeats: more rows than distinct values, and few
    # enough groups that each still gets its own palette colour.
    candidates = [
        c for c in header.columns
        if c not in skip and 1 < header[c].nunique(dropna=True) <= 18
    ]
    if not candidates:
        return None

    prompt = ("Group those categories by a second column?" if exclude
              else "Colour the map by variant group?")
    answer = questionary.select(
        prompt,
        choices=candidates + ["No grouping"],
        default="No grouping",
    ).ask()
    if answer is None:
        raise typer.Exit()
    return None if answer == "No grouping" else answer


@app.command()
def generate(
    project_name: Optional[str] = typer.Argument(None, help="Name for this print project."),
    library_csv: Optional[Path] = typer.Argument(None, help="Path to library CSV file."),
    preset: str = typer.Option("mitomi-ps1.8k", help="Device preset. Use 'custom' to specify --columns/--rows."),
    columns: Optional[int] = typer.Option(None, help="Number of columns (overrides preset)."),
    rows: Optional[int] = typer.Option(None, help="Number of rows (overrides preset)."),
    skip_rows: bool = typer.Option(True, help="Insert blank rows between samples."),
    catalytic_binning: bool = typer.Option(False, help="Sort by catalytic rank before filling."),
    plots: bool = typer.Option(True, help="Generate visualization plots."),
    pinlist: bool = typer.Option(True, help="Write a pinlist CSV alongside the .fld."),
    category_column: Optional[str] = typer.Option(
        None,
        "--category-column",
        help="Library column grouping variants (e.g. a scan name). Each group gets "
             "its own colour on the dashboard map.",
    ),
    group_column: Optional[str] = typer.Option(
        None,
        "--group-column",
        help="Library column giving a second designation for each category, used "
             "to sub-head the sublibraries in the dashboard legend.",
    ),
    seed: int = typer.Option(
        0, "--seed",
        help="Seed for placing variants in the array. The same library, settings "
             "and seed always produce the same layout; change it for a different "
             "arrangement.",
    ),
    refresh: int = typer.Option(
        0, "--refresh",
        help="Seconds between automatic dashboard reloads, so a regenerated sheet "
             "appears without a manual refresh. Off by default.",
    ),
) -> None:
    """
    Generate a Scienion-compatible .fld file from a library CSV.

    When called with no arguments, launches an interactive wizard.
    """
    if project_name is None and library_csv is None:
        _interactive_generate()
        return

    # ── Non-interactive: resolve columns/rows from preset if not overridden ───
    if True:
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

    device = PRESETS.get(preset, {}).get("device") if preset != "custom" else None
    fld_device = PRESETS.get(preset, {}).get("fld_device") if preset != "custom" else None

    _run_pipeline(
        project_name, library_csv, columns, rows, skip_rows, catalytic_binning, plots,
        category_column=category_column, group_column=group_column, refresh=refresh,
        device=device, fld_device=fld_device, seed=seed, write_pinlist=pinlist,
    )


@app.command()
def pinlist(
    fld_file: Optional[Path] = typer.Argument(None, help="Path to the .fld file."),
    library_csv: Optional[Path] = typer.Argument(None, help="Path to library CSV file (Plate/Well/Name columns)."),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path. Default: <fld_name>_pinlist.csv"),
    excel: bool = typer.Option(False, "--excel", help="Export as Excel (.xlsx) instead of CSV."),
    encoding: str = typer.Option("windows-1252", "--encoding", help="Encoding for the .fld file."),
) -> None:
    """
    Convert a .fld file to a pinlist using a library mapping CSV.

    Maps each print position back to variant names using the library CSV.
    When called with no arguments, launches an interactive prompt.
    """
    # ── Interactive prompt ─────────────────────────────────────────────────────
    if fld_file is None and library_csv is None:
        typer.echo("Pinlist conversion wizard…\n")

        raw_fld = questionary.path(
            ".fld file path:",
            validate=lambda p: Path(p).expanduser().is_file() or "File not found",
        ).ask()
        if raw_fld is None:
            raise typer.Exit()
        fld_file = Path(raw_fld).expanduser()

        raw_csv = questionary.path(
            "Library CSV path:",
            validate=lambda p: Path(p).expanduser().is_file() or "File not found",
        ).ask()
        if raw_csv is None:
            raise typer.Exit()
        library_csv = Path(raw_csv).expanduser()

        excel_answer = questionary.confirm("Export as Excel (.xlsx)?", default=False).ask()
        if excel_answer is None:
            raise typer.Exit()
        excel = excel_answer

    else:
        if fld_file is None or library_csv is None:
            typer.echo("Error: provide both FLD_FILE and LIBRARY_CSV, or neither to use the wizard.", err=True)
            raise typer.Exit(code=1)

        if not fld_file.is_file():
            typer.echo(f"Error: '{fld_file}' is not a file.", err=True)
            raise typer.Exit(code=1)

        if not library_csv.is_file():
            typer.echo(f"Error: '{library_csv}' is not a file.", err=True)
            raise typer.Exit(code=1)

    # ── Determine output path ──────────────────────────────────────────────────
    if output is None:
        ext = ".xlsx" if excel else ".csv"
        output = Path.cwd() / f"{fld_file.stem}_pinlist{ext}"
    elif excel and not str(output).endswith(".xlsx"):
        output = Path(str(output).rsplit(".", 1)[0] + ".xlsx")

    typer.echo(f"\nFLD file:    {fld_file}")
    typer.echo(f"Library CSV: {library_csv}")
    typer.echo(f"Output:      {output}\n")

    pinlist_df = fld_to_pinlist(
        fld_path=str(fld_file),
        library_csv=str(library_csv),
        encoding=encoding,
        output=str(output),
    )

    n_variants = pinlist_df[pinlist_df['MutantID'] != 'BLANK']['MutantID'].nunique()
    n_blanks = (pinlist_df['MutantID'] == 'BLANK').sum()
    typer.echo(f"Pinlist generated: {len(pinlist_df)} positions, {n_variants} unique variants, {n_blanks} blanks")
    typer.echo(f"Saved to: {output}")


if __name__ == "__main__":
    app()
