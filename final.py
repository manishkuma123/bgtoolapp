
import os

# Critical: Force minimal memory usage BEFORE imports
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ONNX_DISABLE_EXTERNAL_CUSTOM_OPS"] = "1"
os.environ["U2NET_HOME"] = "/tmp/.u2net"  # Use tmp for model cache

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import subprocess
from rembg import remove, new_session
import uuid
from pathlib import Path
import logging
import json
import uvicorn
import sys
import gc
import io

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("VideoBGRemover")

print("=" * 60, flush=True)
print("🔧 Ultra Lightweight Background Remover (512MB Free Tier)", flush=True)
print("=" * 60, flush=True)

# FastAPI
app = FastAPI(title="Lightweight Background Remover API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

BASE_DIR = Path("/tmp/temp_processing")  # Use /tmp for temporary files
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

job_status = {}

# CRITICAL: Force u2netp model (11MB instead of 176MB)
print("⚙️  Initializing u2netp model (11MB)...", flush=True)
rembg_session = new_session(model_name="u2netp")

# Video limits for 512MB RAM
MAX_VIDEO_DURATION = 10  # seconds
MAX_VIDEO_SIZE_MB = 20   # MB
MAX_RESOLUTION = 480     # pixels (height)
MAX_FPS = 10             # frames per second

def check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      timeout=5)
        return True
    except Exception:
        return False

def get_video_info(video_path: str):
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=10)
        data = json.loads(result.stdout)
        fps_str = data["streams"][0]["r_frame_rate"]
        width = int(data["streams"][0]["width"])
        height = int(data["streams"][0]["height"])
        duration = float(data["format"]["duration"])
        fps = eval(fps_str) if "/" in fps_str else float(fps_str)
        return fps, width, height, duration
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
        return 30, 1280, 720, 10.0

def extract_frames(video_path: str, frames_dir: str, target_fps: float, max_resolution: int):
    """Extract frames at ultra-low resolution for 512MB RAM."""
    try:
        # Ultra-low resolution + limited FPS
        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", f"scale={max_resolution}:-1,fps={target_fps}",
            "-q:v", "5",  # Compression
            f"{frames_dir}/frame_%04d.png",
            "-hide_banner", "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, timeout=60)
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        logger.info(f"✅ Extracted {frame_count} frames at {max_resolution}p @ {target_fps}fps")
        return frame_count
    except subprocess.CalledProcessError as e:
        raise Exception(f"Frame extraction failed: {str(e)}")
    except subprocess.TimeoutExpired:
        raise Exception("Frame extraction timeout - video too large")

def remove_background_from_frames(frames_dir: str, output_dir: str, job_id: str):
    """Process frames one at a time with aggressive cleanup."""
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
        total_frames = len(frame_files)
        
        if total_frames == 0:
            raise Exception("No frames extracted")
        
        logger.info(f"🎬 Processing {total_frames} frames for job {job_id}")

        for idx, frame_file in enumerate(frame_files, 1):
            input_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(output_dir, frame_file)

            try:
                # Read, process, write - one frame at a time
                with open(input_path, "rb") as f:
                    input_data = f.read()

                # Remove background using u2netp session
                output_data = remove(input_data, session=rembg_session)

                with open(output_path, "wb") as out_f:
                    out_f.write(output_data)

                # Aggressive cleanup
                del input_data, output_data
                
                # Delete source frame immediately
                try:
                    os.remove(input_path)
                except:
                    pass

                # Update progress
                job_status[job_id]["processed_frames"] = idx
                job_status[job_id]["progress"] = int((idx / total_frames) * 100)

                # Force garbage collection every 3 frames
                if idx % 3 == 0:
                    gc.collect()
                    logger.info(f"📊 Job {job_id}: {idx}/{total_frames} frames ({job_status[job_id]['progress']}%)")

            except Exception as e:
                logger.error(f"❌ Frame {frame_file} failed: {e}")
                # Continue processing other frames
                continue

        logger.info(f"✅ Background removal completed for job {job_id}")

    except Exception as e:
        logger.error(f"❌ Background removal failed: {e}")
        raise Exception(f"Background removal failed: {str(e)}")

def create_video_from_frames(frames_dir: str, output_path: str, fps: float):
    """Create compressed WebM with transparency."""
    try:
        webm_output = str(output_path).replace(".mp4", ".webm")
        
        # Ultra-compressed WebM
        cmd = [
            "ffmpeg", "-framerate", str(fps),
            "-i", f"{frames_dir}/frame_%04d.png",
            "-c:v", "libvpx-vp9",
            "-pix_fmt", "yuva420p",
            "-auto-alt-ref", "0",
            "-crf", "35",  # Higher = more compression
            "-b:v", "500k",  # Lower bitrate
            "-metadata:s:v:0", 'alpha_mode="1"',
            webm_output,
            "-y",
            "-hide_banner", "-loglevel", "error",
        ]
        subprocess.run(cmd, check=True, timeout=120)
        logger.info(f"🎬 Video created: {webm_output}")
        return webm_output
    except subprocess.CalledProcessError as e:
        raise Exception(f"Video creation failed: {str(e)}")
    except subprocess.TimeoutExpired:
        raise Exception("Video creation timeout")

def cleanup_temp_files(path: Path):
    """Cleanup temporary files."""
    try:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
            logger.info(f"🧹 Cleaned up {path}")
    except Exception as e:
        logger.warning(f"Cleanup warning: {e}")

def process_video(video_path: str, job_id: str):
    """Main processing pipeline with strict memory limits."""
    job_dir = BASE_DIR / job_id
    frames_dir = job_dir / "frames"
    no_bg_dir = job_dir / "no_bg"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"

    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
        no_bg_dir.mkdir(parents=True, exist_ok=True)

        # Get video info
        fps, width, height, duration = get_video_info(str(video_path))
        
        # Enforce limits
        if duration > MAX_VIDEO_DURATION:
            raise Exception(f"Video too long: {duration:.1f}s (max {MAX_VIDEO_DURATION}s for free tier)")
        
        # Calculate target FPS (limit to MAX_FPS)
        target_fps = min(fps, MAX_FPS)
        estimated_frames = int(duration * target_fps)
        
        if estimated_frames > 100:  # Safety limit
            target_fps = 100 / duration
            logger.warning(f"⚠️  Reducing FPS to {target_fps:.1f} to stay within limits")
        
        job_status[job_id].update({
            "original_fps": fps,
            "target_fps": target_fps,
            "original_duration": duration,
            "estimated_frames": int(duration * target_fps),
        })

        # Extract frames
        job_status[job_id]["status"] = "extracting_frames"
        total_frames = extract_frames(str(video_path), str(frames_dir), target_fps, MAX_RESOLUTION)
        job_status[job_id]["total_frames"] = total_frames

        # Cleanup source video immediately
        try:
            if os.path.exists(video_path):
                os.remove(video_path)
        except:
            pass
        gc.collect()

        # Remove background
        job_status[job_id]["status"] = "removing_background"
        remove_background_from_frames(str(frames_dir), str(no_bg_dir), job_id)

        # Cleanup frames
        cleanup_temp_files(frames_dir)
        gc.collect()

        # Create output video
        job_status[job_id]["status"] = "creating_video"
        output_file = create_video_from_frames(str(no_bg_dir), str(output_path), target_fps)

        job_status[job_id].update({
            "status": "completed",
            "output_path": output_file,
            "progress": 100,
        })

        logger.info(f"✅ Job {job_id} completed successfully!")

    except Exception as e:
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)
        logger.error(f"❌ Job {job_id} failed: {e}")

    finally:
        # Final cleanup
        cleanup_temp_files(job_dir)
        gc.collect()

# Routes
@app.get("/")
async def root():
    return {
        "status": "running",
        "version": "4.0-free-tier",
        "model": "u2netp (11MB)",
        "limits": {
            "max_video_duration": f"{MAX_VIDEO_DURATION}s",
            "max_video_size": f"{MAX_VIDEO_SIZE_MB}MB",
            "max_resolution": f"{MAX_RESOLUTION}p",
            "max_fps": MAX_FPS,
        },
        "message": "Optimized for Render free tier (512MB RAM)"
    }

@app.post("/upload-video/")
async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("video/"):
        raise HTTPException(400, "Please upload a valid video file")

    # Check file size
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset
    
    if file_size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"Video too large: {file_size/(1024*1024):.1f}MB (max {MAX_VIDEO_SIZE_MB}MB)")

    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(500, f"Upload failed: {str(e)}")

    job_status[job_id] = {
        "status": "queued",
        "filename": file.filename,
        "progress": 0,
        "processed_frames": 0,
        "file_size_mb": file_size / (1024 * 1024),
    }

    background_tasks.add_task(process_video, str(video_path), job_id)
    
    return {
        "job_id": job_id,
        "message": "Processing started",
        "note": f"Processing at {MAX_RESOLUTION}p @ max {MAX_FPS}fps for free tier"
    }

@app.get("/status/{job_id}")
async def status(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job not found")
    return job_status[job_id]

@app.get("/download/{job_id}")
async def download(job_id: str):
    if job_id not in job_status:
        raise HTTPException(404, "Job not found")
    
    job = job_status[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(400, f"Video not ready. Current status: {job['status']}")
    
    if not os.path.exists(job["output_path"]):
        raise HTTPException(404, "Output file not found")
    
    return FileResponse(
        job["output_path"],
        filename=f"no_bg_{job['filename'].rsplit('.', 1)[0]}.webm",
        media_type="video/webm",
    )

@app.post("/remove-image-background/")
async def remove_image_background(file: UploadFile = File(...)):
    """Remove background from images - works better on free tier."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")
    
    try:
        contents = await file.read()
        
        # Check size
        if len(contents) > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(400, "Image too large (max 5MB)")
        
        # Remove background
        output_data = remove(contents, session=rembg_session)
        
        # Cleanup
        del contents
        gc.collect()
        
        return StreamingResponse(
            io.BytesIO(output_data),
            media_type="image/png",
            headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "model": "u2netp"}

# Main
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    
    # Verify FFmpeg
    if not check_ffmpeg():
        logger.error("❌ FFmpeg not found!")
        sys.exit(1)
    
    print(f"🚀 Starting server on port {port}", flush=True)
    print(f"📊 Limits: {MAX_VIDEO_DURATION}s videos, {MAX_RESOLUTION}p, {MAX_FPS}fps", flush=True)
    print(f"💾 Model: u2netp (11MB)", flush=True)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        timeout_keep_alive=30,
        access_log=False  # Reduce memory
    )