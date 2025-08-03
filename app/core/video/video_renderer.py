"""
Video renderer core logic.

Migrated from src/core/video_renderer.py to provide clean separation
of business logic from framework-specific code.
"""

import os
import re
import subprocess
import asyncio
import concurrent.futures
from PIL import Image
from typing import Optional, List, Union, Dict
import traceback
import sys
import time
import json
import hashlib
from pathlib import Path
import shutil
import tempfile

try:
    import ffmpeg
except ImportError:
    print("Warning: ffmpeg-python not installed. Video combination features will be limited.")
    ffmpeg = None

from .parse_video import (
    get_images_from_video,
    image_with_most_non_black_space
)


class VideoRenderer:
    """Enhanced video renderer with significant performance optimizations."""

    def __init__(self, output_dir="output", print_response=False, use_visual_fix_code=False,
                 max_concurrent_renders=4, enable_caching=True, default_quality="medium",
                 use_gpu_acceleration=False, preview_mode=False):
        """Initialize the enhanced VideoRenderer."""
        self.output_dir = output_dir
        self.print_response = print_response
        self.use_visual_fix_code = use_visual_fix_code
        self.max_concurrent_renders = max_concurrent_renders
        self.enable_caching = enable_caching
        self.default_quality = default_quality
        self.use_gpu_acceleration = use_gpu_acceleration
        self.preview_mode = preview_mode
        
        # Performance monitoring
        self.render_stats = {
            'total_renders': 0,
            'cache_hits': 0,
            'total_time': 0,
            'average_time': 0
        }
        
        # Quality presets for faster rendering
        self.quality_presets = {
            'preview': {'flag': '-ql', 'fps': 15, 'resolution': '480p'},
            'low': {'flag': '-ql', 'fps': 15, 'resolution': '480p'},
            'medium': {'flag': '-qm', 'fps': 30, 'resolution': '720p'},
            'high': {'flag': '-qh', 'fps': 60, 'resolution': '1080p'},
            'production': {'flag': '-qp', 'fps': 60, 'resolution': '1440p'}
        }
        
        # Cache directory for rendered scenes
        self.cache_dir = os.path.join(output_dir, '.render_cache')
        if enable_caching:
            os.makedirs(self.cache_dir, exist_ok=True)
        
        # Thread pool for concurrent operations
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent_renders)    
        
    def render_scene(self, code: str, file_prefix: str, curr_scene: int, 
                    curr_version: int, code_dir: str, media_dir: str, 
                    use_visual_fix_code=False, visual_self_reflection_func=None, 
                    banned_reasonings=None, scene_trace_id=None, topic=None, 
                    session_id=None, code_generator=None, 
                    scene_implementation=None, description=None, 
                    scene_outline=None) -> tuple:
        """Render a scene with intelligent error handling and code generation fixes."""
        
        start_time = time.time()
        current_code = code
        quality = self.default_quality
        
        # Check cache first
        cached_video = self._is_cached(current_code, quality)
        if cached_video:
            # Copy cached video to expected location
            expected_path = self._get_expected_video_path(file_prefix, curr_scene, curr_version, media_dir)
            os.makedirs(os.path.dirname(expected_path), exist_ok=True)
            shutil.copy2(cached_video, expected_path)
            
            elapsed = time.time() - start_time
            print(f"Scene {curr_scene} rendered from cache in {elapsed:.2f}s")
            return current_code, None

        # Optimize manim command for speed
        file_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
        
        # Write optimized code file
        self._write_code_file(file_path, current_code)
        
        # Build optimized manim command
        manim_cmd = self._build_optimized_command(file_path, media_dir, quality)
        
        max_retries = 3
        retries = 0
        while retries < max_retries:
            try:
                print(f"🎬 Rendering scene {curr_scene} (quality: {quality}, attempt: {retries + 1})")
                
                # Execute manim with optimizations
                result = self._run_manim_optimized(manim_cmd, file_path)

                if result.returncode != 0:
                    raise Exception(result.stderr)

                # Find the rendered video
                video_path = self._find_rendered_video(file_prefix, curr_scene, curr_version, media_dir)
                
                # Save to cache
                self._save_to_cache(current_code, quality, video_path)

                elapsed = time.time() - start_time
                self.render_stats['total_renders'] += 1
                self.render_stats['total_time'] += elapsed
                self.render_stats['average_time'] = self.render_stats['total_time'] / self.render_stats['total_renders']
                
                print(f"Scene {curr_scene} rendered successfully in {elapsed:.2f}s")
                
                return current_code, None

            except Exception as e:
                print(f"Render attempt {retries + 1} failed: {e}")
                
                # Save error log
                error_log_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}_error_{retries}.log")
                self._write_error_log(error_log_path, str(e), retries)
                
                # Try to fix the code if we have a code generator
                if code_generator and scene_implementation and retries < max_retries - 1:
                    print(f"🔧 Attempting to fix code using CodeGenerator (attempt {retries + 1})")
                    try:
                        fixed_code, fix_log = code_generator.fix_code_errors(
                            implementation_plan=scene_implementation,
                            code=current_code,
                            error=str(e),
                            scene_trace_id=scene_trace_id,
                            topic=topic,
                            scene_number=curr_scene,
                            session_id=session_id
                        )
                        
                        if fixed_code and fixed_code != current_code:
                            print(f"✨ Code fix generated, updating for next attempt")
                            current_code = fixed_code
                            curr_version += 1
                            
                            # Update file path and write fixed code
                            file_path = os.path.join(code_dir, f"{file_prefix}_scene{curr_scene}_v{curr_version}.py")
                            self._write_code_file(file_path, current_code)
                            
                            # Update manim command for new file
                            manim_cmd = self._build_optimized_command(file_path, media_dir, quality)
                        else:
                            print(f"⚠️ Code generator returned same or empty code, doing standard retry")
                    except Exception as fix_error:
                        print(f"❌ Code fix attempt failed: {fix_error}")
                
                retries += 1
                if retries < max_retries:
                    time.sleep(1)  # Brief delay before retry
                else:
                    return current_code, str(e)

        return current_code, f"Failed after {max_retries} attempts"    
    def _get_code_hash(self, code: str) -> str:
        """Generate hash for code to enable caching."""
        return hashlib.md5(code.encode()).hexdigest()

    def _get_cache_path(self, code_hash: str, quality: str) -> str:
        """Get cache file path for given code hash and quality."""
        return os.path.join(self.cache_dir, f"{code_hash}_{quality}.mp4")

    def _is_cached(self, code: str, quality: str) -> Optional[str]:
        """Check if rendered video exists in cache."""
        if not self.enable_caching:
            return None
        
        code_hash = self._get_code_hash(code)
        cache_path = self._get_cache_path(code_hash, quality)
        
        if os.path.exists(cache_path):
            print(f"Cache hit for code hash {code_hash[:8]}...")
            self.render_stats['cache_hits'] += 1
            return cache_path
        return None

    def _save_to_cache(self, code: str, quality: str, video_path: str):
        """Save rendered video to cache."""
        if not self.enable_caching or not os.path.exists(video_path):
            return
        
        code_hash = self._get_code_hash(code)
        cache_path = self._get_cache_path(code_hash, quality)
        
        try:
            shutil.copy2(video_path, cache_path)
            print(f"Cached render for hash {code_hash[:8]}...")
        except Exception as e:
            print(f"Warning: Could not cache render: {e}")

    def _build_optimized_command(self, file_path: str, media_dir: str, quality: str) -> List[str]:
        """Build optimized manim command with performance flags."""
        quality_preset = self.quality_presets.get(quality, self.quality_presets['medium'])
        
        cmd = [
            "manim",
            "render",
            quality_preset['flag'],  # Quality setting
            file_path,
            "--media_dir", media_dir,
            "--fps", str(quality_preset['fps'])
        ]
        
        # Add caching option (only disable if needed)
        if not self.enable_caching:
            cmd.append("--disable_caching")
        
        # Add GPU acceleration if available and enabled
        if self.use_gpu_acceleration:
            cmd.extend(["--renderer", "opengl"])
        
        # Preview mode optimizations
        if self.preview_mode or quality == 'preview':
            cmd.extend([
                "--save_last_frame",  # Only render final frame for quick preview
                "--write_to_movie"    # Skip unnecessary file operations
            ])
        
        return cmd

    def _run_manim_optimized(self, cmd: List[str], file_path: str) -> subprocess.CompletedProcess:
        """Run manim command with optimizations."""
        env = os.environ.copy()
        
        # Optimize environment for performance
        env.update({
            'MANIM_DISABLE_CACHING': 'false' if self.enable_caching else 'true',
            'MANIM_VERBOSITY': 'WARNING',  # Reduce log verbosity
            'OMP_NUM_THREADS': str(os.cpu_count()),  # Use all CPU cores
            'MANIM_RENDERER_TIMEOUT': '300'  # 5 minute timeout
        })
        
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=300  # 5 minute timeout
        )

    def _write_code_file(self, file_path: str, code: str):
        """Write code file with optimizations."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Add optimization hints to the code
        optimized_code = self._optimize_code_for_rendering(code)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(optimized_code)    
            
    def _optimize_code_for_rendering(self, code: str) -> str:
        """Add optimization hints to Manim code."""
        optimizations = [
            "",
            "# Manim rendering optimizations",
            "from manim import config",
            "config.frame_rate = 30  # Balanced frame rate",
            "config.pixel_height = 720  # Optimized resolution",
            "config.pixel_width = 1280",
            ""
        ]
        
        # Find the end of manim imports specifically
        lines = code.split('\n')
        manim_import_end = 0
        
        for i, line in enumerate(lines):
            # Look for manim-related imports
            if (line.strip().startswith('from manim') or 
                line.strip().startswith('import manim') or
                line.strip().startswith('from manim_')):
                manim_import_end = i + 1
        
        # If no manim imports found, look for the end of all imports
        if manim_import_end == 0:
            for i, line in enumerate(lines):
                if (line.strip().startswith(('from ', 'import ')) and 
                    not line.strip().startswith('#')):
                    manim_import_end = i + 1
        
        # Insert optimization code after manim imports
        lines[manim_import_end:manim_import_end] = optimizations
        
        return '\n'.join(lines)

    def _write_error_log(self, file_path: str, error: str, attempt: int):
        """Write error log."""
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        log_content = f"[{timestamp}] Attempt {attempt + 1}: {error}\n"
        
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(log_content)

    def _get_expected_video_path(self, file_prefix: str, scene: int, version: int, media_dir: str) -> str:
        """Get expected path for rendered video."""
        return os.path.join(
            media_dir, "videos", f"{file_prefix}_scene{scene}_v{version}", 
            "1080p60", f"{file_prefix}_scene{scene}_v{version}.mp4"
        )

    def _find_rendered_video(self, file_prefix: str, scene: int, version: int, media_dir: str) -> str:
        """Find the rendered video file."""
        video_dir = os.path.join(media_dir, "videos", f"{file_prefix}_scene{scene}_v{version}")
        
        # Look in quality-specific subdirectories
        for quality_dir in ["1080p60", "720p30", "480p15"]:
            search_dir = os.path.join(video_dir, quality_dir)
            if os.path.exists(search_dir):
                for file in os.listdir(search_dir):
                    if file.endswith('.mp4'):
                        return os.path.join(search_dir, file)
        
        raise FileNotFoundError(f"No rendered video found for scene {scene} version {version}")

    def combine_videos(self, topic: str) -> str:
        """Combine videos for a topic."""
        file_prefix = re.sub(r'[^a-z0-9_]+', '_', topic.lower())
        
        print(f"🎬 Starting video combination for topic: {topic}")
        
        # Prepare paths
        video_output_dir = os.path.join(self.output_dir, file_prefix)
        output_video_path = os.path.join(video_output_dir, f"{file_prefix}_combined.mp4")
        
        # Check if already exists
        if os.path.exists(output_video_path):
            print(f"Combined video already exists at {output_video_path}")
            return output_video_path
        
        # Get scene information
        scene_videos = self._gather_scene_files(file_prefix)
        
        if not scene_videos:
            raise ValueError("No scene videos found to combine")
        
        print(f"📹 Found {len(scene_videos)} scene videos to combine")
        
        try:
            # Simple concatenation for now
            self._simple_video_combination(scene_videos, output_video_path)
            
            print(f"🎉 Video combination completed")
            print(f"📁 Output: {output_video_path}")
            
            return output_video_path
            
        except Exception as e:
            print(f"❌ Error in video combination: {e}")
            traceback.print_exc()
            raise

    def _gather_scene_files(self, file_prefix: str) -> List[str]:
        """Gather scene video files."""
        search_path = os.path.join(self.output_dir, file_prefix, "media", "videos")
        
        if not os.path.exists(search_path):
            return []
        
        scene_videos = []
        
        # Find all scene directories
        for item in os.listdir(search_path):
            if item.startswith(f"{file_prefix}_scene") and os.path.isdir(os.path.join(search_path, item)):
                scene_dir = os.path.join(search_path, item)
                
                # Look for video files in quality subdirectories
                for quality_dir in ["1080p60", "720p30", "480p15"]:
                    quality_path = os.path.join(scene_dir, quality_dir)
                    if os.path.exists(quality_path):
                        for filename in os.listdir(quality_path):
                            if filename.endswith('.mp4'):
                                scene_videos.append(os.path.join(quality_path, filename))
                                break
                        break
        
        # Sort by scene number
        scene_videos.sort(key=lambda x: self._extract_scene_number(x))
        return scene_videos

    def _extract_scene_number(self, video_path: str) -> int:
        """Extract scene number from video path."""
        match = re.search(r'scene(\d+)', video_path)
        return int(match.group(1)) if match else 0

    def _simple_video_combination(self, scene_videos: List[str], output_path: str):
        """Simple video combination using ffmpeg."""
        if not scene_videos:
            raise ValueError("No videos to combine")
        
        # Create a temporary file list for ffmpeg
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for video in scene_videos:
                f.write(f"file '{os.path.abspath(video)}'\n")
            filelist_path = f.name
        
        try:
            # Use ffmpeg to concatenate videos
            cmd = [
                'ffmpeg',
                '-f', 'concat',
                '-safe', '0',
                '-i', filelist_path,
                '-c', 'copy',
                '-y',  # Overwrite output file
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ Videos combined successfully")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ FFmpeg error: {e.stderr}")
            raise
        finally:
            # Clean up temporary file
            try:
                os.unlink(filelist_path)
            except:
                pass