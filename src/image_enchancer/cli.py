import os
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional, Tuple
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn, DownloadColumn, TransferSpeedColumn

from image_enchancer.models import MODEL_REGISTRY, resolve_model_path, get_cache_dir, download_file
from image_enchancer.pipeline import ImageEnhancementPipeline, EnhancementConfig

console = Console()


def is_url(target: str) -> bool:
    """Checks whether a string target represents an HTTP or HTTPS URL."""
    return target.startswith("http://") or target.startswith("https://")


def send_desktop_notification(title: str, message: str, icon_path: Optional[str] = None):
    """Sends a desktop notification using notify-send with an optional image thumbnail."""
    try:
        cmd = ["notify-send", title, message, "--app-name=Image Enhancer"]
        if icon_path and Path(icon_path).exists():
            cmd.extend(["-i", str(icon_path)])
        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def fetch_source_image(target: str) -> Tuple[Path, str, bool]:
    """
    Resolves local path or downloads remote URL into a local file.
    Returns (local_path, display_name, is_temp).
    """
    if not is_url(target):
        path = Path(target)
        if not path.exists():
            console.print(f"[bold red]Error:[/] Input file does not exist: {target}")
            sys.exit(1)
        return path, path.name, False

    parsed = urllib.parse.urlparse(target)
    url_path = Path(parsed.path)
    filename = url_path.name if url_path.name else "remote_image.png"

    # Ensure valid extension
    if not Path(filename).suffix.lower() in [".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"]:
        filename += ".png"

    temp_dir = Path(tempfile.gettempdir()) / "image_enchancer_downloads"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_path = temp_dir / filename

    console.print(f"[bold cyan]Downloading remote image:[/] {target}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Fetching image[/]"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=console,
    ) as progress:
        req = urllib.request.Request(
            target,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) ImageEnhancer/1.0"},
        )
        with urllib.request.urlopen(req) as response, open(temp_path, "wb") as out_file:
            total = int(response.headers.get("Content-Length", 0))
            task = progress.add_task("download", total=total)
            chunk_size = 32 * 1024
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                progress.update(task, advance=len(chunk))

    return temp_path, filename, True


@click.group()
def cli():
    """Image Quality Maximizer: Neural Super-Resolution and Face Restoration Pipeline."""
    pass


@cli.command("enhance")
@click.argument("input_target", type=str)
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
@click.option("--notify/--no-notify", default=True, show_default=True,
              help="Send desktop notification with image thumbnail upon completion.")
@click.option("--device", type=str, default=None,
              help="Compute device ('cpu', 'cuda', etc.). Defaults to auto-detection.")
def enhance(
    input_target: str,
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
    notify: bool,
    device: Optional[str],
):
    """Enhance a local image, directory, or remote image URL to maximum quality."""
    run_enhance_pipeline(
        input_target=input_target,
        output_path=output_path,
        model=model,
        scale=scale,
        face_restore=face_restore,
        face_fidelity=face_fidelity,
        tile_size=tile_size,
        tile_pad=tile_pad,
        denoise=denoise,
        clarity=clarity,
        sharpen=sharpen,
        color_match=color_match,
        notify=notify,
        device=device,
    )


def run_enhance_pipeline(
    input_target: str,
    output_path: Optional[Path] = None,
    model: str = "realesrgan-x4plus",
    scale: Optional[float] = None,
    face_restore: bool = True,
    face_fidelity: float = 0.85,
    tile_size: int = 512,
    tile_pad: int = 32,
    denoise: float = 0.0,
    clarity: float = 1.2,
    sharpen: float = 0.35,
    color_match: bool = True,
    notify: bool = True,
    device: Optional[str] = None,
):
    """Underlying orchestrator for both structured and direct CLI invocations."""
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
        f"[bold cyan]Image Quality Maximizer[/]\n"
        f"Target: [yellow]{input_target}[/] | Model: [green]{model}[/] | Face Restoration: [green]{face_restore}[/]",
        border_style="cyan",
    ))

    with console.status("[bold yellow]Initializing neural models...[/]", spinner="dots"):
        pipeline = ImageEnhancementPipeline(config)

    # Check if directory
    is_dir = Path(input_target).is_dir() if not is_url(input_target) else False
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}

    if is_dir:
        dir_path = Path(input_target)
        files = sorted([p for p in dir_path.iterdir() if p.suffix.lower() in valid_exts])
        if not files:
            console.print(f"[red]No valid images found in directory: {input_target}[/]")
            sys.exit(1)
        if output_path is None:
            output_path = dir_path / "enhanced"
        targets = [(f, f.name, False) for f in files]
    else:
        local_path, display_name, is_temp = fetch_source_image(input_target)
        targets = [(local_path, display_name, is_temp)]
        if output_path is None:
            stem = Path(display_name).stem
            suffix = Path(display_name).suffix or ".png"
            output_path = Path.cwd() / f"{stem}_enhanced{suffix}"

    results_table = Table(title="Processing Summary", show_header=True, header_style="bold magenta")
    results_table.add_column("Filename", style="dim")
    results_table.add_column("Input Dimensions")
    results_table.add_column("Output Dimensions")
    results_table.add_column("Faces Restored", justify="right")
    results_table.add_column("Processing Time", justify="right")

    for local_file, display_name, is_temp in targets:
        if len(targets) == 1 and not output_path.is_dir() and output_path.suffix:
            dest_file = output_path
        else:
            stem = Path(display_name).stem
            suffix = Path(display_name).suffix or ".png"
            dest_file = output_path / f"{stem}_enhanced{suffix}"

        console.print(f"[bold blue]Processing:[/] {display_name} -> {dest_file.name}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}[/]"),
            BarColumn(),
            TextColumn("[bold green]{task.percentage:>3.0f}%[/]"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Preparing {display_name}...", total=100)

            def update_progress(completed: int, total: int, status_text: str = ""):
                pct = int((completed / max(total, 1)) * 100)
                desc = status_text if status_text else f"Tile {completed}/{total}"
                progress.update(task, completed=pct, description=desc)

            try:
                telemetry = pipeline.process_file(
                    input_path=local_file,
                    output_path=dest_file,
                    progress_callback=update_progress,
                )
                progress.update(task, completed=100, description=f"Finalizing {display_name}...")

                w_orig, h_orig = telemetry["original_dimensions"]
                w_out, h_out = telemetry["output_dimensions"]
                results_table.add_row(
                    display_name,
                    f"{w_orig}x{h_orig}",
                    f"{w_out}x{h_out}",
                    str(telemetry["faces_restored"]),
                    f"{telemetry['elapsed_seconds']:.2f}s",
                )
                console.print(f"[bold green]Saved:[/] {dest_file.resolve()}")

                # Trigger desktop notification
                if notify:
                    send_desktop_notification(
                        title="Image Enhancement Complete",
                        message=f"{dest_file.name} ({w_out}x{h_out}) saved in {telemetry['elapsed_seconds']:.1f}s",
                        icon_path=str(dest_file),
                    )
            except Exception as ex:
                console.print(f"[bold red]Error processing {display_name}:[/] {ex}")
            finally:
                if is_temp and local_file.exists():
                    try:
                        local_file.unlink()
                    except Exception:
                        pass

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


@click.command()
@click.argument("target", type=str)
@click.argument("output", type=click.Path(path_type=Path), required=False, default=None)
@click.option("-m", "--model", default="realesrgan-x4plus", show_default=True,
              help="Neural super-resolution model preset or path to custom weights.")
@click.option("-s", "--scale", type=float, default=None,
              help="Target upscale factor (e.g. 2.0, 4.0).")
@click.option("--face-restore/--no-face-restore", default=True, show_default=True,
              help="Enable/disable GFPGAN face restoration.")
@click.option("--anime", is_flag=True, default=False,
              help="Shortcut for anime/digital art mode (--model realesrgan-anime --no-face-restore).")
@click.option("--notify/--no-notify", default=True, show_default=True,
              help="Send desktop notification with image thumbnail upon completion.")
def direct_main_command(
    target: str,
    output: Optional[Path],
    model: str,
    scale: Optional[float],
    face_restore: bool,
    anime: bool,
    notify: bool,
):
    """
    Direct single-command image enhancer.
    Accepts a local image file, directory, or remote image URL.

    Usage:
      image-enhance <image_path_or_url> [output_path]
    """
    if anime:
        model = "realesrgan-anime"
        face_restore = False

    run_enhance_pipeline(
        input_target=target,
        output_path=output,
        model=model,
        scale=scale,
        face_restore=face_restore,
        notify=notify,
    )


def direct_main():
    direct_main_command()


def main():
    cli()


if __name__ == "__main__":
    main()
