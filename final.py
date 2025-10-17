from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import subprocess
from rembg import remove
from PIL import Image
import uuid
from pathlib import Path
import logging
import json
import os
import uvicorn


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Video Background Remover API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path("temp_processing")
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

job_status = {}

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      stdout=subprocess.PIPE, 
                      stderr=subprocess.PIPE, 
                      check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def get_video_info(video_path: str):
    """Get video information using ffprobe"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,width,height',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        
        fps_str = data['streams'][0]['r_frame_rate']
        width = int(data['streams'][0]['width'])
        height = int(data['streams'][0]['height'])
        duration = float(data['format']['duration'])
        
        # Convert fps fraction to decimal
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        logger.info(f"Video info - FPS: {fps}, Resolution: {width}x{height}, Duration: {duration}s")
        return fps, width, height, duration
    except Exception as e:
        logger.warning(f"Could not get video info: {e}")
        return 30, 1920, 1080, 10.0

def extract_frames(video_path: str, frames_dir: str, fps: float):
    """Extract frames from video at original FPS"""
    try:
        logger.info(f"Extracting frames from {video_path} at {fps} fps")
        cmd = [
            'ffmpeg', '-i', video_path,
            '-vf', f'fps={fps}',
            f'{frames_dir}/frame_%06d.png',
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        
        frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        logger.info(f"Extracted {frame_count} frames")
        return frame_count
    except subprocess.CalledProcessError as e:
        logger.error(f"Frame extraction failed: {e.stderr.decode()}")
        raise Exception(f"Frame extraction failed: {e.stderr.decode()}")

def remove_background_from_frames(frames_dir: str, output_dir: str, job_id: str):
    """Remove background from all frames with transparency"""
    try:
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        total_frames = len(frame_files)
        
        logger.info(f"Processing {total_frames} frames for job {job_id}")
        
        for idx, frame_file in enumerate(frame_files, 1):
            input_path = os.path.join(frames_dir, frame_file)
            output_path = os.path.join(output_dir, frame_file)
            
            with open(input_path, 'rb') as f:
                input_data = f.read()
            
            # Remove background with transparency
            output_data = remove(input_data)
            
            # Ensure RGBA format
            img = Image.open(__import__('io').BytesIO(output_data))
    
            if img.mode != 'RGBA':
                img = img.convert('RGBA')
        
            img.save(output_path, 'PNG')

            job_status[job_id]['processed_frames'] = idx
            job_status[job_id]['progress'] = int((idx / total_frames) * 100)
            
            if idx % 10 == 0:
                logger.info(f"Job {job_id}: Processed {idx}/{total_frames} frames")
        
        logger.info(f"Background removal completed for job {job_id}")
    except Exception as e:
        logger.error(f"Background removal failed: {str(e)}")
        raise Exception(f"Background removal failed: {str(e)}")

def create_video_from_frames(frames_dir: str, output_path: str, fps: float, width: int, height: int):
    """Create video from frames with transparency support using WebM"""
    try:
        logger.info(f"Creating WebM video with transparency: {fps}fps, {width}x{height}")
 
        webm_output = str(output_path).replace('.mp4', '.webm')
        cmd = [
            'ffmpeg', '-framerate', str(fps),
            '-i', f'{frames_dir}/frame_%06d.png',
            '-c:v', 'libvpx-vp9',
            '-pix_fmt', 'yuva420p',
            '-auto-alt-ref', '0',
            '-crf', '15',
            '-b:v', '2000k',
            '-metadata:s:v:0', 'alpha_mode="1"',
            webm_output,
            '-y',
            '-hide_banner', '-loglevel', 'error'
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        logger.info(f"WebM video created successfully at {webm_output} with TRANSPARENT background")
        return webm_output
    except subprocess.CalledProcessError as e:
        logger.error(f"Video creation failed: {e.stderr.decode()}")
        raise Exception(f"Video creation failed: {e.stderr.decode()}")

def verify_alpha_channel(video_path: str):
    """Verify if the video has alpha channel"""
    try:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=pix_fmt',
            '-of', 'json',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        pix_fmt = data['streams'][0]['pix_fmt']
        has_alpha = 'yuva' in pix_fmt or 'rgba' in pix_fmt or 'gbra' in pix_fmt
        logger.info(f"Video pixel format: {pix_fmt}, Has alpha: {has_alpha}")
        return has_alpha
    except Exception as e:
        logger.warning(f"Could not verify alpha channel: {e}")
        return False

def cleanup_temp_files(job_dir: Path):
    """Clean up temporary files"""
    try:
        if job_dir.exists():
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up temporary files for {job_dir}")
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")

def process_video(video_path: str, job_id: str):
    """Main video processing function with transparency"""
    job_dir = BASE_DIR / job_id
    frames_dir = job_dir / "frames"
    no_bg_dir = job_dir / "no_bg"
    output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
    try:
        frames_dir.mkdir(parents=True, exist_ok=True)
        no_bg_dir.mkdir(parents=True, exist_ok=True)
        
        # Get original video info
        logger.info(f"Analyzing video: {video_path}")
        fps, width, height, duration = get_video_info(str(video_path))
        job_status[job_id]['original_fps'] = fps
        job_status[job_id]['original_width'] = width
        job_status[job_id]['original_height'] = height
        job_status[job_id]['original_duration'] = duration
        
        # Extract frames at ORIGINAL FPS
        job_status[job_id]['status'] = 'extracting_frames'
        total_frames = extract_frames(str(video_path), str(frames_dir), fps)
        job_status[job_id]['total_frames'] = total_frames
        
        # Remove backgrounds with transparency
        job_status[job_id]['status'] = 'removing_backgrounds'
        remove_background_from_frames(str(frames_dir), str(no_bg_dir), job_id)
        
        # Create video with transparency
        job_status[job_id]['status'] = 'creating_video'
        output_file = create_video_from_frames(str(no_bg_dir), str(output_path), fps, width, height)

        # Verify alpha channel
        has_alpha = verify_alpha_channel(output_file)
        job_status[job_id]['has_alpha'] = has_alpha

        job_status[job_id]['status'] = 'completed'
        job_status[job_id]['output_path'] = str(output_file)
        job_status[job_id]['progress'] = 100
        
        logger.info(f"Job {job_id} completed - FPS: {fps}, Resolution: {width}x{height}, Duration: {duration}s, Transparent BG: {has_alpha}, Format: WebM")
        
    except Exception as e:
        job_status[job_id]['status'] = 'failed'
        job_status[job_id]['error'] = str(e)
        logger.error(f"Job {job_id} failed: {str(e)}")
    finally:
        cleanup_temp_files(job_dir)
        if os.path.exists(video_path):
            os.remove(video_path)

@app.on_event("startup")
async def startup_event():
    """Check dependencies on startup"""
    if not check_ffmpeg():
        logger.error("FFmpeg is not installed!")
        raise RuntimeError("FFmpeg is required but not found in system PATH")
    logger.info("FFmpeg found - API ready with transparent background support")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Video Background Remover API",
        "version": "2.0",
        "status": "running",
        "features": ["Transparent Background", "Original Quality", "Original FPS", "Alpha Channel Verification"]
    }

@app.post("/upload-video/")
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload video and start processing"""
    
    if not file.content_type or not file.content_type.startswith('video/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload a video file."
        )
    
    job_id = str(uuid.uuid4())
    video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Video uploaded: {file.filename} (Job ID: {job_id})")
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to save uploaded file: {str(e)}"
        )
    
    job_status[job_id] = {
        'status': 'queued',
        'filename': file.filename,
        'total_frames': 0,
        'processed_frames': 0,
        'progress': 0,
        'original_fps': 0,
        'original_width': 0,
        'original_height': 0,
        'original_duration': 0,
        'has_alpha': False
    }
    
    background_tasks.add_task(process_video, str(video_path), job_id)
    
    return {
        'job_id': job_id,
        'filename': file.filename,
        'message': 'Video uploaded successfully. Processing started with transparent background.'
    }

@app.post("/remove-image-background/")
async def remove_image_background(file: UploadFile = File(...)):
    """Remove background from image and return PNG with transparency"""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail="Invalid file type. Please upload an image file."
        )
    
    try:
        # Read the uploaded image
        contents = await file.read()
        
        # Remove background
        logger.info(f"Processing image: {file.filename}")
        output_data = remove(contents)
        
        # Convert to PIL Image to ensure RGBA
        img = Image.open(__import__('io').BytesIO(output_data))
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # Save to bytes
        img_byte_arr = __import__('io').BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        logger.info(f"Image processed successfully: {file.filename}")
        
        # Return the processed image
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            img_byte_arr,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=no_bg_{file.filename.rsplit('.', 1)[0]}.png"
            }
        )
        
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Image processing failed: {str(e)}"
        )

@app.get("/status/{job_id}")
async def get_status(job_id: str):
    """Get job processing status"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]

@app.get("/download/{job_id}")
async def download_video(job_id: str):
    """Download processed video with transparent background"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    
    if status['status'] != 'completed':
        raise HTTPException(
            status_code=400, 
            detail=f"Video not ready. Current status: {status['status']}"
        )
    
    output_path = status.get('output_path')
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Processed video not found")
    
    filename = f"no_bg_{status['filename']}"
    
    return FileResponse(
        path=output_path,
        filename=filename.replace('.mp4', '.webm'),
        media_type="video/webm",
        headers={"Content-Disposition": f"attachment; filename={filename.replace('.mp4', '.webm')}"}
    )

@app.delete("/cleanup/{job_id}")
async def cleanup_job(job_id: str):
    """Delete processed video and job data"""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    status = job_status[job_id]
    output_path = status.get('output_path')
    
    if output_path and os.path.exists(output_path):
        os.remove(output_path)
    
    del job_status[job_id]
    
    return {"message": "Job cleaned up successfully"}

# if __name__ == "__main__":
#     import uvicorn
#      port = int(os.environ.get("PORT", 8000))
#     # uvicorn.run(app, host="0.0.0.0", port=8002)
#     uvicorn.run("final:app", host="0.0.0.0", port=port)

if __name__ == "__main__":
   
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Starting FastAPI on {port}")
    uvicorn.run("final:app", host="0.0.0.0", port=port)

# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 8000))
#     uvicorn.run("final:app", host="0.0.0.0", port=port)