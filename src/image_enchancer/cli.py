import sys
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from image_enchancer.models import MODEL_REGISTRY, resolve_model_path, get_cache_dir, download_file
from image_enchancer.pipeline import ImageEnhancementPipeline, EnhancementConfig

console = Console()


@click.group()
def cli():
    """Image Quality Maximizer: Neural Super-Resolution and Face Restoration Pipeline."""
    pass


@cli.command("enhance")
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", "output_path", type=click.Path(path_type=Path), default=None,
              help="Output file path or destination directory.")
@click.option("-m", "--model", default="realesrgan-x4plus", show_default=True,
              help="Neural super-resolution model preset or path to custom weights (.pth, .safetensors).")
@click.option("-s", "--scale", type=float, default=None,
              help="Target upscale factor (e.g. 2.0, 4.0). Defaults to native model scale.")
@click.option("--face-restore/--no-face-restore", default=True, show_default=True,
              help="Enable GFPGAN blind face restoration on detected human faces.")
@click.option("--face-fidelity", type=float, default=0.85, show_default=True,
              help="Restoration fidelity weight (0.0 = original face, 1.0 = full generative reconstruction).")
@click.option("--tile-size", type=int, default=512, show_default=True,
              help="Tile dimension for memory-bounded processing. Set to 0 to disable tiling.")
@click.option("--tile-pad", type=int, default=32, show_default=True,
              help="Contextual overlap padding in pixels for seamless tile stitching.")
@click.option("--denoise", type=float, default=0.0, show_default=True,
              help="Pre-processing bilateral denoising strength (0.0 to 1.0).")
@click.option("--clarity", type=float, default=1.2, show_default=True,
              help="Luminance CLAHE micro-contrast enhancement strength (0.0 to disable).")
@click.option("--sharpen", type=float, default=0.35, show_default=True,
              help="Edge-preserving high-frequency unsharp masking strength (0.0 to disable).")
@click.option("--color-match/--no-color-match", default=True, show_default=True,
              help="Enforce statistical color distribution matching against input image.")
@click.option("--device", type=str, default=None,
              help="Compute device ('cpu', 'cuda', etc.). Defaults to auto-detection.")
def enhance(
    input_path: Path,
    output_path: Optional[Path],
    model: str,
    scale: Optional[float],
    face_restore: bool,
    face_fidelity: float,
    tile_size: int,
    tile_pad: int,
    denoise: float,
    clarity: float,
    sharpen: float,
    color_match: bool,
    device: Optional[str],
):
    """Enhance a single image or an entire directory of images to maximum quality."""
    config = EnhancementConfig(
        model_name=model,
        scale=scale,
        face_restore=face_restore,
        face_fidelity=face_fidelity,
        tile_size=tile_size,
        tile_pad=tile_pad,
        denoise_strength=denoise,
        clarity_strength=clarity,
        sharpen_strength=sharpen,
        color_match=color_match,
        device=device,
    )

    console.print(Panel.fit(
        f"[bold cyan]Image Quality Maximizer Pipeline[/]\n"
        f"Model: [green]{model}[/] | Face Restoration: [green]{face_restore}[/] | Scale: [green]{scale or 'Native'}[/]",
        border_style="cyan",
    ))

    with console.status("[bold yellow]Initializing neural models...[/]", spinner="dots"):
        pipeline = ImageEnhancementPipeline(config)

    # Collect input files
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
    if input_path.is_dir():
        files = sorted([p for p in input_path.iterdir() if p.suffix.lower() in valid_exts])
        if not files:
            console.print(f"[red]No valid images found in directory: {input_path}[/]")
            sys.exit(1)
        if output_path is None:
            output_path = input_path / "enhanced"
    else:
        files = [input_path]
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_enhanced{input_path.suffix}"

    results_table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    results_table.add_column("Filename", style="dim")
    results_table.add_column("Input Dimensions")
    results_table.add_column("Output Dimensions")
    results_table.add_column("Faces Restored", justify="right")
    results_table.add_column("Processing Time", justify="right")

    for file_item in files:
        if len(files) == 1 and not output_path.is_dir() and output_path.suffix:
            dest_file = output_path
        else:
            dest_file = output_path / f"{file_item.stem}_enhanced{file_item.suffix}"

        console.print(f"[bold blue]Processing:[/] {file_item.name} -> {dest_file.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Enhancing {file_item.name}", total=100)

            def update_progress(completed: int, total: int):
                pct = int((completed / max(total, 1)) * 100)
                progress.update(task, completed=pct)

            try:
                telemetry = pipeline.process_file(
                    input_path=file_item,
                    output_path=dest_file,
                    progress_callback=update_progress,
                )
                progress.update(task, completed=100)

                w_orig, h_orig = telemetry["original_dimensions"]
                w_out, h_out = telemetry["output_dimensions"]
                results_table.add_row(
                    file_item.name,
                    f"{w_orig}x{h_orig}",
                    f"{w_out}x{h_out}",
                    str(telemetry["faces_restored"]),
                    f"{telemetry['elapsed_seconds']:.2f}s",
                )
            except Exception as ex:
                console.print(f"[bold red]Error processing {file_item.name}:[/] {ex}")

    console.print()
    console.print(results_table)


@cli.group("models")
def models_group():
    """Manage preconfigured models."""
    pass


@models_group.command("list")
def list_models():
    """List all supported preset models and their local availability status."""
    table = Table(title="Registered Enhancement Models", show_header=True, header_style="bold cyan")
    table.add_column("Preset Key", style="bold")
    table.add_column("Type")
    table.add_column("Scale")
    table.add_column("Cached Locally", justify="center")
    table.add_column("Description")

    cache_dir = get_cache_dir()
    local_models_dir = Path.cwd() / "models"

    for key, meta in MODEL_REGISTRY.items():
        fname = meta["filename"]
        local_exists = (local_models_dir / fname).exists() or (cache_dir / fname).exists()
        cached_str = "[green]Yes[/]" if local_exists else "[yellow]No (auto-downloads)[/]"
        table.add_row(
            key,
            meta["type"].upper(),
            meta["scale"] + "x",
            cached_str,
            meta["description"],
        )

    console.print(table)


@models_group.command("download")
@click.argument("model_name", required=False, default="all")
def download_preset_models(model_name: str):
    """Pre-download weights for offline usage."""
    targets = MODEL_REGISTRY.keys() if model_name.lower() == "all" else [model_name.lower()]
    for name in targets:
        if name not in MODEL_REGISTRY:
            console.print(f"[red]Unknown model preset: {name}[/]")
            continue
        meta = MODEL_REGISTRY[name]
        dest = get_cache_dir() / meta["filename"]
        if dest.exists() and dest.stat().st_size > 0:
            console.print(f"[green]Model '{name}' already present at {dest}[/]")
        else:
            download_file(meta["url"], dest, description=f"Downloading {meta['filename']}")
            console.print(f"[green]Successfully fetched {meta['filename']}[/]")


@cli.command("ui")
@click.option("--port", type=int, default=7860, show_default=True, help="Port to bind the web interface.")
@click.option("--host", type=str, default="127.0.0.1", show_default=True, help="Host interface to bind.")
def launch_ui(port: int, host: str):
    """Launch the interactive web UI in your browser."""
    from app import build_app
    demo = build_app()
    console.print(f"[bold green]Starting Image Enhancer Web UI on http://{host}:{port}[/]")
    demo.launch(server_name=host, server_port=port, inbrowser=False)


def main():
    cli()


if __name__ == "__main__":
    main()
