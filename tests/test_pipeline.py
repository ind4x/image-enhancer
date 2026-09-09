import cv2
import numpy as np
import pytest
from pathlib import Path
from image_enchancer.models import resolve_model_path, load_neural_model
from image_enchancer.postprocess import PostProcessor
from image_enchancer.super_resolution import SuperResolutionEngine
from image_enchancer.face_restore import FaceRestorationEngine
from image_enchancer.pipeline import ImageEnhancementPipeline, EnhancementConfig


def test_post_processor_clahe():
    """Verify CLAHE preserves image dimensions and data range."""
    img = np.random.randint(0, 256, (120, 160, 3), dtype=np.uint8)
    enhanced = PostProcessor.apply_clahe(img, clip_limit=1.5)
    assert enhanced.shape == img.shape
    assert enhanced.dtype == np.uint8


def test_post_processor_unsharp_mask():
    """Verify unsharp mask preserves shape and respects deadband threshold."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    # Flat image should have zero sharpening delta due to threshold
    sharpened = PostProcessor.apply_unsharp_mask(img, strength=0.5, threshold=5)
    assert np.array_equal(img, sharpened)


def test_post_processor_color_matching():
    """Verify color distribution matching between two images."""
    source = np.full((100, 100, 3), 50, dtype=np.uint8)
    reference = np.full((50, 50, 3), 200, dtype=np.uint8)
    matched = PostProcessor.match_color_distribution(source, reference)
    assert matched.shape == source.shape
    # Matched mean should be close to reference mean
    assert abs(int(matched.mean()) - 200) < 10


def test_super_resolution_tiled():
    """Verify tiled super-resolution engine on CPU with lightweight anime model."""
    model = load_neural_model("realesrgan-anime")
    engine = SuperResolutionEngine(model)
    
    # Input 64x64
    test_img = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
    # Use small tile size to force tiling code path
    out = engine.upscale(test_img, tile_size=32, tile_pad=8)
    assert out.shape == (256, 256, 3)
    assert out.dtype == np.uint8


def test_face_detector():
    """Verify YuNet detector initializes and executes without exceptions."""
    face_engine = FaceRestorationEngine()
    dummy_img = np.zeros((300, 300, 3), dtype=np.uint8)
    restored, count = face_engine.restore_faces(dummy_img)
    assert restored.shape == dummy_img.shape
    assert count == 0


def test_pipeline_end_to_end(tmp_path: Path):
    """Verify full end-to-end pipeline execution from file to file."""
    # Create test input image
    test_input = tmp_path / "input_sample.png"
    test_output = tmp_path / "output_sample.png"
    
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    cv2.circle(img, (40, 40), 20, (0, 255, 0), -1)
    cv2.imwrite(str(test_input), img)

    config = EnhancementConfig(
        model_name="realesrgan-anime",
        scale=4.0,
        face_restore=False,
        tile_size=64,
        tile_pad=8,
        clarity_strength=1.0,
        sharpen_strength=0.3,
        color_match=True,
    )
    pipeline = ImageEnhancementPipeline(config)
    telemetry = pipeline.process_file(test_input, test_output)

    assert test_output.exists()
    assert telemetry["original_dimensions"] == (80, 80)
    assert telemetry["output_dimensions"] == (320, 320)
    assert telemetry["effective_scale"] == 4.0


def test_vulkan_availability():
    """Verify Vulkan availability detection runs without exceptions."""
    from image_enchancer.vulkan import is_vulkan_available
    available = is_vulkan_available()
    assert isinstance(available, bool)


def test_pipeline_engine_config():
    """Verify engine selection in pipeline configuration."""
    config = EnhancementConfig(engine="cpu")
    assert config.engine == "cpu"
    pipeline = ImageEnhancementPipeline(config)
    assert pipeline.config.engine == "cpu"

