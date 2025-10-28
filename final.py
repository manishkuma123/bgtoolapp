# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import os
# import subprocess
# from rembg import remove
# from PIL import Image
# import uuid
# from pathlib import Path
# import logging
# import json
# import uvicorn
# import sys

# # Configure logging FIRST
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     stream=sys.stdout
# )
# logger = logging.getLogger(__name__)


# print("="*60, flush=True)
# print("🔧 Initializing Video Background Remover API", flush=True)
# print("="*60, flush=True)

# app = FastAPI(title="Video Background Remover API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = Path("temp_processing")
# UPLOAD_DIR = BASE_DIR / "uploads"
# OUTPUT_DIR = BASE_DIR / "outputs"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# job_status = {}

# def check_ffmpeg():
#     """Check if ffmpeg is installed"""
#     try:
#         result = subprocess.run(
#             ['ffmpeg', '-version'], 
#             stdout=subprocess.PIPE, 
#             stderr=subprocess.PIPE,
#             timeout=5
#         )
#         return result.returncode == 0
#     except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
#         return False

# def get_video_info(video_path: str):
#     """Get video information using ffprobe"""
#     try:
#         cmd = [
#             'ffprobe', '-v', 'error',
#             '-select_streams', 'v:0',
#             '-show_entries', 'stream=r_frame_rate,width,height',
#             '-show_entries', 'format=duration',
#             '-of', 'json',
#             video_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         data = json.loads(result.stdout)
        
#         fps_str = data['streams'][0]['r_frame_rate']
#         width = int(data['streams'][0]['width'])
#         height = int(data['streams'][0]['height'])
#         duration = float(data['format']['duration'])
        
#         if '/' in fps_str:
#             num, den = map(int, fps_str.split('/'))
#             fps = num / den
#         else:
#             fps = float(fps_str)
        
#         logger.info(f"Video info - FPS: {fps}, Resolution: {width}x{height}, Duration: {duration}s")
#         return fps, width, height, duration
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
#         return 30, 1920, 1080, 10.0

# def extract_frames(video_path: str, frames_dir: str, fps: float):
#     """Extract frames from video at original FPS"""
#     try:
#         logger.info(f"Extracting frames from {video_path} at {fps} fps")
#         cmd = [
#             'ffmpeg', '-i', video_path,
#             '-vf', f'fps={fps}',
#             f'{frames_dir}/frame_%06d.png',
#             '-hide_banner', '-loglevel', 'error'
#         ]
#         subprocess.run(cmd, check=True, capture_output=True)
        
#         frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
#         logger.info(f"Extracted {frame_count} frames")
#         return frame_count
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Frame extraction failed: {e.stderr.decode()}")
#         raise Exception(f"Frame extraction failed: {e.stderr.decode()}")

# def remove_background_from_frames(frames_dir: str, output_dir: str, job_id: str):
#     """Remove background from all frames with transparency"""
#     try:
#         frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
#         total_frames = len(frame_files)
        
#         logger.info(f"Processing {total_frames} frames for job {job_id}")
        
#         for idx, frame_file in enumerate(frame_files, 1):
#             input_path = os.path.join(frames_dir, frame_file)
#             output_path = os.path.join(output_dir, frame_file)
            
#             with open(input_path, 'rb') as f:
#                 input_data = f.read()
            
#             output_data = remove(input_data)
            
#             img = Image.open(__import__('io').BytesIO(output_data))
    
#             if img.mode != 'RGBA':
#                 img = img.convert('RGBA')
        
#             img.save(output_path, 'PNG')

#             job_status[job_id]['processed_frames'] = idx
#             job_status[job_id]['progress'] = int((idx / total_frames) * 100)
            
#             if idx % 10 == 0:
#                 logger.info(f"Job {job_id}: Processed {idx}/{total_frames} frames")
        
#         logger.info(f"Background removal completed for job {job_id}")
#     except Exception as e:
#         logger.error(f"Background removal failed: {str(e)}")
#         raise Exception(f"Background removal failed: {str(e)}")

# def create_video_from_frames(frames_dir: str, output_path: str, fps: float, width: int, height: int):
#     """Create video from frames with transparency support using WebM"""
#     try:
#         logger.info(f"Creating WebM video with transparency: {fps}fps, {width}x{height}")
 
#         webm_output = str(output_path).replace('.mp4', '.webm')
#         cmd = [
#             'ffmpeg', '-framerate', str(fps),
#             '-i', f'{frames_dir}/frame_%06d.png',
#             '-c:v', 'libvpx-vp9',
#             '-pix_fmt', 'yuva420p',
#             '-auto-alt-ref', '0',
#             '-crf', '15',
#             '-b:v', '2000k',
#             '-metadata:s:v:0', 'alpha_mode="1"',
#             webm_output,
#             '-y',
#             '-hide_banner', '-loglevel', 'error'
#         ]
#         subprocess.run(cmd, check=True, capture_output=True)
#         logger.info(f"WebM video created successfully at {webm_output} with TRANSPARENT background")
#         return webm_output
#     except subprocess.CalledProcessError as e:
#         logger.error(f"Video creation failed: {e.stderr.decode()}")
#         raise Exception(f"Video creation failed: {e.stderr.decode()}")

# def verify_alpha_channel(video_path: str):
#     """Verify if the video has alpha channel"""
#     try:
#         cmd = [
#             'ffprobe', '-v', 'error',
#             '-select_streams', 'v:0',
#             '-show_entries', 'stream=pix_fmt',
#             '-of', 'json',
#             video_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         data = json.loads(result.stdout)
#         pix_fmt = data['streams'][0]['pix_fmt']
#         has_alpha = 'yuva' in pix_fmt or 'rgba' in pix_fmt or 'gbra' in pix_fmt
#         logger.info(f"Video pixel format: {pix_fmt}, Has alpha: {has_alpha}")
#         return has_alpha
#     except Exception as e:
#         logger.warning(f"Could not verify alpha channel: {e}")
#         return False

# def cleanup_temp_files(job_dir: Path):
#     """Clean up temporary files"""
#     try:
#         if job_dir.exists():
#             shutil.rmtree(job_dir)
#             logger.info(f"Cleaned up temporary files for {job_dir}")
#     except Exception as e:
#         logger.error(f"Cleanup failed: {str(e)}")

# def process_video(video_path: str, job_id: str):
#     """Main video processing function with transparency"""
#     job_dir = BASE_DIR / job_id
#     frames_dir = job_dir / "frames"
#     no_bg_dir = job_dir / "no_bg"
#     output_path = OUTPUT_DIR / f"{job_id}.mp4"
    
#     try:
#         frames_dir.mkdir(parents=True, exist_ok=True)
#         no_bg_dir.mkdir(parents=True, exist_ok=True)
        
#         logger.info(f"Analyzing video: {video_path}")
#         fps, width, height, duration = get_video_info(str(video_path))
#         job_status[job_id]['original_fps'] = fps
#         job_status[job_id]['original_width'] = width
#         job_status[job_id]['original_height'] = height
#         job_status[job_id]['original_duration'] = duration
        
#         job_status[job_id]['status'] = 'extracting_frames'
#         total_frames = extract_frames(str(video_path), str(frames_dir), fps)
#         job_status[job_id]['total_frames'] = total_frames
        
#         job_status[job_id]['status'] = 'removing_backgrounds'
#         remove_background_from_frames(str(frames_dir), str(no_bg_dir), job_id)
        
#         job_status[job_id]['status'] = 'creating_video'
#         output_file = create_video_from_frames(str(no_bg_dir), str(output_path), fps, width, height)

#         has_alpha = verify_alpha_channel(output_file)
#         job_status[job_id]['has_alpha'] = has_alpha

#         job_status[job_id]['status'] = 'completed'
#         job_status[job_id]['output_path'] = str(output_file)
#         job_status[job_id]['progress'] = 100
        
#         logger.info(f"Job {job_id} completed - FPS: {fps}, Resolution: {width}x{height}, Duration: {duration}s, Transparent BG: {has_alpha}, Format: WebM")
        
#     except Exception as e:
#         job_status[job_id]['status'] = 'failed'
#         job_status[job_id]['error'] = str(e)
#         logger.error(f"Job {job_id} failed: {str(e)}")
#     finally:
#         cleanup_temp_files(job_dir)
#         if os.path.exists(video_path):
#             os.remove(video_path)

# @app.on_event("startup")
# async def startup_event():
#     """Check dependencies on startup - NON-BLOCKING"""
#     try:
#         logger.info("=" * 50)
#         logger.info("Starting Video Background Remover API")
#         logger.info("=" * 50)
        
#         ffmpeg_available = check_ffmpeg()
        
#         if not ffmpeg_available:
#             logger.warning("⚠️  FFmpeg is not installed! Video processing will not work.")
#             logger.warning("⚠️  Image processing will still work.")
#         else:
#             logger.info("✓ FFmpeg found - Video processing available")
        
#         logger.info("✓ API ready with transparent background support")
#     except Exception as e:
#         logger.error(f"Startup event error (non-fatal): {e}")

# @app.get("/")
# async def root():
#     """Root endpoint"""
#     return {
#         "message": "Video Background Remover API",
#         "version": "2.0",
#         "status": "running",
#         "features": ["Transparent Background", "Original Quality", "Original FPS", "Alpha Channel Verification"]
#     }

# @app.get("/health")
# async def health():
#     """Health check endpoint"""
#     return {
#         "status": "healthy",
#         "ffmpeg_available": check_ffmpeg()
#     }

# @app.post("/upload-video/")
# async def upload_video(
#     background_tasks: BackgroundTasks,
#     file: UploadFile = File(...)
# ):
#     """Upload video and start processing"""
    
#     if not file.content_type or not file.content_type.startswith('video/'):
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid file type. Please upload a video file."
#         )
    
#     job_id = str(uuid.uuid4())
#     video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
#     try:
#         with open(video_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
#         logger.info(f"Video uploaded: {file.filename} (Job ID: {job_id})")
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, 
#             detail=f"Failed to save uploaded file: {str(e)}"
#         )
    
#     job_status[job_id] = {
#         'status': 'queued',
#         'filename': file.filename,
#         'total_frames': 0,
#         'processed_frames': 0,
#         'progress': 0,
#         'original_fps': 0,
#         'original_width': 0,
#         'original_height': 0,
#         'original_duration': 0,
#         'has_alpha': False
#     }
    
#     background_tasks.add_task(process_video, str(video_path), job_id)
    
#     return {
#         'job_id': job_id,
#         'filename': file.filename,
#         'message': 'Video uploaded successfully. Processing started with transparent background.'
#     }

# @app.post("/remove-image-background/")
# async def remove_image_background(file: UploadFile = File(...)):
#     """Remove background from image and return PNG with transparency"""
    
#     if not file.content_type or not file.content_type.startswith('image/'):
#         raise HTTPException(
#             status_code=400, 
#             detail="Invalid file type. Please upload an image file."
#         )
    
#     try:
#         contents = await file.read()
        
#         logger.info(f"Processing image: {file.filename}")
#         output_data = remove(contents)
        
#         img = Image.open(__import__('io').BytesIO(output_data))
#         if img.mode != 'RGBA':
#             img = img.convert('RGBA')
        
#         img_byte_arr = __import__('io').BytesIO()
#         img.save(img_byte_arr, format='PNG')
#         img_byte_arr.seek(0)
        
#         logger.info(f"Image processed successfully: {file.filename}")
        
#         from fastapi.responses import StreamingResponse
#         return StreamingResponse(
#             img_byte_arr,
#             media_type="image/png",
#             headers={
#                 "Content-Disposition": f"attachment; filename=no_bg_{file.filename.rsplit('.', 1)[0]}.png"
#             }
#         )
        
#     except Exception as e:
#         logger.error(f"Image processing failed: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Image processing failed: {str(e)}"
#         )

# @app.get("/status/{job_id}")
# async def get_status(job_id: str):
#     """Get job processing status"""
#     if job_id not in job_status:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     return job_status[job_id]

# @app.get("/download/{job_id}")
# async def download_video(job_id: str):
#     """Download processed video with transparent background"""
#     if job_id not in job_status:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     status = job_status[job_id]
    
#     if status['status'] != 'completed':
#         raise HTTPException(
#             status_code=400, 
#             detail=f"Video not ready. Current status: {status['status']}"
#         )
    
#     output_path = status.get('output_path')
#     if not output_path or not os.path.exists(output_path):
#         raise HTTPException(status_code=404, detail="Processed video not found")
    
#     filename = f"no_bg_{status['filename']}"
    
#     return FileResponse(
#         path=output_path,
#         filename=filename.replace('.mp4', '.webm'),
#         media_type="video/webm",
#         headers={"Content-Disposition": f"attachment; filename={filename.replace('.mp4', '.webm')}"}
#     )

# @app.delete("/cleanup/{job_id}")
# async def cleanup_job(job_id: str):
#     """Delete processed video and job data"""
#     if job_id not in job_status:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     status = job_status[job_id]
#     output_path = status.get('output_path')
    
#     if output_path and os.path.exists(output_path):
#         os.remove(output_path)
    
#     del job_status[job_id]
    
#     return {"message": "Job cleaned up successfully"}

# if __name__ == "__main__":
#     try:
#         port = int(os.environ.get("PORT", 10000))
        
#         print("=" * 60, flush=True)
#         print(f"🚀 Starting FastAPI Server", flush=True)
#         print(f"📡 Host: 0.0.0.0", flush=True)
#         print(f"🔌 Port: {port}", flush=True)
#         print(f"🌐 Environment: {'Production (Render)' if os.environ.get('PORT') else 'Development'}", flush=True)
#         print("=" * 60, flush=True)
#         sys.stdout.flush()
        
        
#         uvicorn.run(
#             app,
#             host="0.0.0.0",
#             port=port,
#             log_level="info",
#             access_log=True,
#             timeout_keep_alive=30
#         )
#     except Exception as e:
#         print(f"❌ FATAL ERROR: {e}", flush=True)
#         logger.exception("Failed to start server")
#         sys.exit(1)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Disable GPU detection
# os.environ["OMP_NUM_THREADS"] = "1"       # Limit CPU threads
# os.environ["ONNX_DISABLE_EXTERNAL_CUSTOM_OPS"] = "1"  # Lighter ONNX init

# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import FileResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import os
# import subprocess
# from rembg import remove, new_session
# from PIL import Image
# import uuid
# from pathlib import Path
# import logging
# import json
# import uvicorn
# import sys
# import gc
# import io  # needed for StreamingResponse

# # ==========================================================
# # CONFIG & LOGGING
# # ==========================================================
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     stream=sys.stdout
# )
# logger = logging.getLogger("VideoBGRemover")

# print("=" * 60, flush=True)
# print("🔧 Initializing Video Background Remover API (Memory-Optimized)", flush=True)
# print("=" * 60, flush=True)

# # ==========================================================
# # FASTAPI APP SETUP
# # ==========================================================
# app = FastAPI(title="Memory-Safe Video Background Remover API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = Path("temp_processing")
# UPLOAD_DIR = BASE_DIR / "uploads"
# OUTPUT_DIR = BASE_DIR / "outputs"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# job_status = {}

# # ==========================================================
# # GLOBAL REMBG SESSION
# # ==========================================================
# rembg_session = new_session()

# # ==========================================================
# # UTILITIES
# # ==========================================================
# def check_ffmpeg():
#     try:
#         subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
#         return True
#     except Exception:
#         return False


# def get_video_info(video_path: str):
#     try:
#         cmd = [
#             'ffprobe', '-v', 'error',
#             '-select_streams', 'v:0',
#             '-show_entries', 'stream=r_frame_rate,width,height',
#             '-show_entries', 'format=duration',
#             '-of', 'json',
#             video_path
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         data = json.loads(result.stdout)
#         fps_str = data['streams'][0]['r_frame_rate']
#         width = int(data['streams'][0]['width'])
#         height = int(data['streams'][0]['height'])
#         duration = float(data['format']['duration'])

#         fps = eval(fps_str) if '/' in fps_str else float(fps_str)
#         return fps, width, height, duration
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
#         return 30, 1280, 720, 10.0


# def extract_frames(video_path: str, frames_dir: str, fps: float):
#     """Extract frames from video at given FPS (downscaled for safety)."""
#     try:
#         cmd = [
#             'ffmpeg', '-i', video_path,
#             '-vf', f'scale=1280:-1,fps={fps}',
#             f'{frames_dir}/frame_%06d.png',
#             '-hide_banner', '-loglevel', 'error'
#         ]
#         subprocess.run(cmd, check=True)
#         frame_count = len([f for f in os.listdir(frames_dir) if f.endswith('.png')])
#         logger.info(f"Extracted {frame_count} frames")
#         return frame_count
#     except subprocess.CalledProcessError as e:
#         raise Exception(f"Frame extraction failed: {e.stderr.decode()}")


# def remove_background_from_frames(frames_dir: str, output_dir: str, job_id: str):
#     """Memory-safe background removal from frames using global rembg session."""
#     try:
#         frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
#         total_frames = len(frame_files)
#         logger.info(f"Processing {total_frames} frames for job {job_id} (memory-optimized)")

#         for idx, frame_file in enumerate(frame_files, 1):
#             input_path = os.path.join(frames_dir, frame_file)
#             output_path = os.path.join(output_dir, frame_file)

#             try:
#                 with open(input_path, 'rb') as f:
#                     input_data = f.read()

#                 output_data = remove(input_data, session=rembg_session)

#                 with open(output_path, 'wb') as out_f:
#                     out_f.write(output_data)

#                 # Free memory
#                 del input_data, output_data
#                 gc.collect()

#                 # Delete processed frame to save disk
#                 os.remove(input_path)

#                 # Progress
#                 job_status[job_id]['processed_frames'] = idx
#                 job_status[job_id]['progress'] = int((idx / total_frames) * 100)

#                 if idx % 10 == 0:
#                     logger.info(f"Job {job_id}: {idx}/{total_frames} frames processed")

#             except Exception as e:
#                 logger.error(f"Frame {frame_file} failed: {e}")
#                 continue

#         logger.info(f"✅ Background removal completed for job {job_id}")

#     except Exception as e:
#         logger.error(f"Background removal failed: {e}")
#         raise Exception(f"Background removal failed: {e}")


# def create_video_from_frames(frames_dir: str, output_path: str, fps: float, width: int, height: int):
#     """Create WebM video with transparency."""
#     try:
#         webm_output = str(output_path).replace('.mp4', '.webm')
#         cmd = [
#             'ffmpeg', '-framerate', str(fps),
#             '-i', f'{frames_dir}/frame_%06d.png',
#             '-c:v', 'libvpx-vp9',
#             '-pix_fmt', 'yuva420p',
#             '-auto-alt-ref', '0',
#             '-crf', '18',
#             '-b:v', '1500k',
#             '-metadata:s:v:0', 'alpha_mode="1"',
#             webm_output,
#             '-y',
#             '-hide_banner', '-loglevel', 'error'
#         ]
#         subprocess.run(cmd, check=True)
#         logger.info(f"🎬 Video created successfully: {webm_output}")
#         return webm_output
#     except subprocess.CalledProcessError as e:
#         raise Exception(f"Video creation failed: {e.stderr.decode()}")


# def cleanup_temp_files(path: Path):
#     """Safe cleanup."""
#     try:
#         if path.exists():
#             shutil.rmtree(path)
#             logger.info(f"🧹 Cleaned up {path}")
#     except Exception as e:
#         logger.error(f"Cleanup failed: {e}")


# def process_video(video_path: str, job_id: str):
#     """Main background removal pipeline (optimized)."""
#     job_dir = BASE_DIR / job_id
#     frames_dir = job_dir / "frames"
#     no_bg_dir = job_dir / "no_bg"
#     output_path = OUTPUT_DIR / f"{job_id}.mp4"

#     try:
#         frames_dir.mkdir(parents=True, exist_ok=True)
#         no_bg_dir.mkdir(parents=True, exist_ok=True)

#         fps, width, height, duration = get_video_info(str(video_path))
#         job_status[job_id].update({
#             'original_fps': fps, 'original_width': width,
#             'original_height': height, 'original_duration': duration
#         })

#         job_status[job_id]['status'] = 'extracting_frames'
#         total_frames = extract_frames(str(video_path), str(frames_dir), fps)
#         job_status[job_id]['total_frames'] = total_frames

#         job_status[job_id]['status'] = 'removing_background'
#         remove_background_from_frames(str(frames_dir), str(no_bg_dir), job_id)

#         cleanup_temp_files(frames_dir)

#         job_status[job_id]['status'] = 'creating_video'
#         output_file = create_video_from_frames(str(no_bg_dir), str(output_path), fps, width, height)

#         job_status[job_id].update({
#             'status': 'completed',
#             'output_path': output_file,
#             'progress': 100
#         })

#         logger.info(f"✅ Job {job_id} completed successfully!")

#     except Exception as e:
#         job_status[job_id]['status'] = 'failed'
#         job_status[job_id]['error'] = str(e)
#         logger.error(f"❌ Job {job_id} failed: {e}")

#     finally:
#         cleanup_temp_files(job_dir)
#         if os.path.exists(video_path):
#             os.remove(video_path)

# # ==========================================================
# # ROUTES
# # ==========================================================
# @app.get("/")
# async def root():
#     return {"status": "running", "version": "3.0", "optimized": True}


# @app.post("/upload-video/")
# async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("video/"):
#         raise HTTPException(400, "Upload a valid video file")

#     job_id = str(uuid.uuid4())
#     video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     job_status[job_id] = {
#         'status': 'queued',
#         'filename': file.filename,
#         'progress': 0,
#         'processed_frames': 0
#     }

#     background_tasks.add_task(process_video, str(video_path), job_id)
#     return {"job_id": job_id, "message": "Processing started"}


# @app.get("/status/{job_id}")
# async def status(job_id: str):
#     if job_id not in job_status:
#         raise HTTPException(404, "Job not found")
#     return job_status[job_id]


# @app.get("/download/{job_id}")
# async def download(job_id: str):
#     if job_id not in job_status:
#         raise HTTPException(404, "Job not found")
#     job = job_status[job_id]
#     if job['status'] != 'completed':
#         raise HTTPException(400, f"Not ready, current: {job['status']}")
#     return FileResponse(
#         job['output_path'],
#         filename=f"no_bg_{job['filename'].rsplit('.', 1)[0]}.webm",
#         media_type="video/webm"
#     )


# @app.post("/remove-image-background/")
# async def remove_image_background(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(400, "Upload an image file")
#     try:
#         contents = await file.read()
#         output_data = remove(contents, session=rembg_session)
#         return StreamingResponse(
#             io.BytesIO(output_data),
#             media_type="image/png",
#             headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"}
#         )
#     except Exception as e:
#         raise HTTPException(500, f"Image processing failed: {e}")


# # ==========================================================
# # MAIN
# # ==========================================================
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     print(f"🚀 Starting FastAPI server on port {port}", flush=True)
#     uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
# import os

# # Critical: Force minimal memory usage BEFORE imports
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["ONNX_DISABLE_EXTERNAL_CUSTOM_OPS"] = "1"

# from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
# from fastapi.responses import FileResponse, StreamingResponse
# from fastapi.middleware.cors import CORSMiddleware
# import shutil
# import subprocess
# from rembg import remove, new_session
# import uuid
# from pathlib import Path
# import logging
# import json
# import uvicorn
# import sys
# import gc
# import io

# # Logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     stream=sys.stdout,
# )
# logger = logging.getLogger("VideoBGRemover")

# print("=" * 60, flush=True)
# print("🔧 Ultra Memory-Optimized Background Remover API", flush=True)
# print("=" * 60, flush=True)

# # FastAPI
# app = FastAPI(title="Memory-Safe Background Remover API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# BASE_DIR = Path("temp_processing")
# UPLOAD_DIR = BASE_DIR / "uploads"
# OUTPUT_DIR = BASE_DIR / "outputs"

# UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# job_status = {}

# # Use smallest model available
# rembg_session = new_session("u2netp")  # 11MB model vs 176MB

# def check_ffmpeg():
#     try:
#         subprocess.run(["ffmpeg", "-version"], 
#                       stdout=subprocess.PIPE, 
#                       stderr=subprocess.PIPE, 
#                       timeout=5)
#         return True
#     except Exception:
#         return False

# def get_video_info(video_path: str):
#     try:
#         cmd = [
#             "ffprobe", "-v", "error",
#             "-select_streams", "v:0",
#             "-show_entries", "stream=r_frame_rate,width,height",
#             "-show_entries", "format=duration",
#             "-of", "json",
#             video_path,
#         ]
#         result = subprocess.run(cmd, capture_output=True, text=True, check=True)
#         data = json.loads(result.stdout)
#         fps_str = data["streams"][0]["r_frame_rate"]
#         width = int(data["streams"][0]["width"])
#         height = int(data["streams"][0]["height"])
#         duration = float(data["format"]["duration"])
#         fps = eval(fps_str) if "/" in fps_str else float(fps_str)
#         return fps, width, height, duration
#     except Exception as e:
#         logger.warning(f"Could not get video info: {e}")
#         return 30, 1280, 720, 10.0

# def extract_frames(video_path: str, frames_dir: str, fps: float):
#     """Extract frames at 480p for memory efficiency."""
#     try:
#         # Lower resolution = less memory per frame
#         cmd = [
#             "ffmpeg", "-i", video_path,
#             "-vf", f"scale=480:-1,fps={min(fps, 15)}",  # Cap at 15fps
#             f"{frames_dir}/frame_%06d.png",
#             "-hide_banner", "-loglevel", "error",
#         ]
#         subprocess.run(cmd, check=True)
#         frame_count = len([f for f in os.listdir(frames_dir) if f.endswith(".png")])
#         logger.info(f"Extracted {frame_count} frames at 480p")
#         return frame_count
#     except subprocess.CalledProcessError as e:
#         raise Exception(f"Frame extraction failed: {e.stderr.decode()}")

# def remove_background_from_frames(frames_dir: str, output_dir: str, job_id: str):
#     """Ultra memory-safe background removal."""
#     try:
#         frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".png")])
#         total_frames = len(frame_files)
#         logger.info(f"Processing {total_frames} frames for job {job_id}")

#         for idx, frame_file in enumerate(frame_files, 1):
#             input_path = os.path.join(frames_dir, frame_file)
#             output_path = os.path.join(output_dir, frame_file)

#             try:
#                 # Process one frame at a time
#                 with open(input_path, "rb") as f:
#                     input_data = f.read()

#                 output_data = remove(input_data, session=rembg_session)

#                 with open(output_path, "wb") as out_f:
#                     out_f.write(output_data)

#                 # Aggressive cleanup
#                 del input_data, output_data
#                 gc.collect()

#                 # Delete source immediately
#                 os.remove(input_path)

#                 job_status[job_id]["processed_frames"] = idx
#                 job_status[job_id]["progress"] = int((idx / total_frames) * 100)

#                 if idx % 5 == 0:
#                     logger.info(f"Job {job_id}: {idx}/{total_frames} frames")

#             except Exception as e:
#                 logger.error(f"Frame {frame_file} failed: {e}")
#                 continue

#         logger.info(f"✅ Background removal completed for job {job_id}")

#     except Exception as e:
#         logger.error(f"Background removal failed: {e}")
#         raise Exception(f"Background removal failed: {e}")

# def create_video_from_frames(frames_dir: str, output_path: str, fps: float, width: int, height: int):
#     """Create WebM with transparency."""
#     try:
#         webm_output = str(output_path).replace(".mp4", ".webm")
#         cmd = [
#             "ffmpeg", "-framerate", str(fps),
#             "-i", f"{frames_dir}/frame_%06d.png",
#             "-c:v", "libvpx-vp9",
#             "-pix_fmt", "yuva420p",
#             "-auto-alt-ref", "0",
#             "-crf", "30",  # Lower quality = less memory
#             "-b:v", "1000k",
#             "-metadata:s:v:0", 'alpha_mode="1"',
#             webm_output,
#             "-y",
#             "-hide_banner", "-loglevel", "error",
#         ]
#         subprocess.run(cmd, check=True)
#         logger.info(f"🎬 Video created: {webm_output}")
#         return webm_output
#     except subprocess.CalledProcessError as e:
#         raise Exception(f"Video creation failed: {e.stderr.decode()}")

# def cleanup_temp_files(path: Path):
#     try:
#         if path.exists():
#             shutil.rmtree(path)
#             logger.info(f"🧹 Cleaned up {path}")
#     except Exception as e:
#         logger.error(f"Cleanup failed: {e}")

# def process_video(video_path: str, job_id: str):
#     """Main processing pipeline."""
#     job_dir = BASE_DIR / job_id
#     frames_dir = job_dir / "frames"
#     no_bg_dir = job_dir / "no_bg"
#     output_path = OUTPUT_DIR / f"{job_id}.mp4"

#     try:
#         frames_dir.mkdir(parents=True, exist_ok=True)
#         no_bg_dir.mkdir(parents=True, exist_ok=True)

#         fps, width, height, duration = get_video_info(str(video_path))
#         job_status[job_id].update({
#             "original_fps": fps, 
#             "original_width": width,
#             "original_height": height, 
#             "original_duration": duration,
#         })

#         job_status[job_id]["status"] = "extracting_frames"
#         total_frames = extract_frames(str(video_path), str(frames_dir), fps)
#         job_status[job_id]["total_frames"] = total_frames

#         # Cleanup source video immediately
#         if os.path.exists(video_path):
#             os.remove(video_path)
#         gc.collect()

#         job_status[job_id]["status"] = "removing_background"
#         remove_background_from_frames(str(frames_dir), str(no_bg_dir), job_id)

#         cleanup_temp_files(frames_dir)
#         gc.collect()

#         job_status[job_id]["status"] = "creating_video"
#         output_file = create_video_from_frames(str(no_bg_dir), str(output_path), fps, width, height)

#         job_status[job_id].update({
#             "status": "completed",
#             "output_path": output_file,
#             "progress": 100,
#         })

#         logger.info(f"✅ Job {job_id} completed successfully!")

#     except Exception as e:
#         job_status[job_id]["status"] = "failed"
#         job_status[job_id]["error"] = str(e)
#         logger.error(f"❌ Job {job_id} failed: {e}")

#     finally:
#         cleanup_temp_files(job_dir)
#         gc.collect()

# # Routes
# @app.get("/")
# async def root():
#     return {
#         "status": "running", 
#         "version": "3.1", 
#         "memory_optimized": True,
#         "model": "u2netp (11MB)"
#     }

# @app.post("/upload-video/")
# async def upload_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("video/"):
#         raise HTTPException(400, "Upload a valid video file")

#     job_id = str(uuid.uuid4())
#     video_path = UPLOAD_DIR / f"{job_id}_{file.filename}"

#     with open(video_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     job_status[job_id] = {
#         "status": "queued",
#         "filename": file.filename,
#         "progress": 0,
#         "processed_frames": 0,
#     }

#     background_tasks.add_task(process_video, str(video_path), job_id)
#     return {"job_id": job_id, "message": "Processing started"}

# @app.get("/status/{job_id}")
# async def status(job_id: str):
#     if job_id not in job_status:
#         raise HTTPException(404, "Job not found")
#     return job_status[job_id]

# @app.get("/download/{job_id}")
# async def download(job_id: str):
#     if job_id not in job_status:
#         raise HTTPException(404, "Job not found")
#     job = job_status[job_id]
#     if job["status"] != "completed":
#         raise HTTPException(400, f"Not ready, current: {job['status']}")
#     return FileResponse(
#         job["output_path"],
#         filename=f"no_bg_{job['filename'].rsplit('.', 1)[0]}.webm",
#         media_type="video/webm",
#     )

# @app.post("/remove-image-background/")
# async def remove_image_background(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(400, "Upload an image file")
#     try:
#         contents = await file.read()
#         output_data = remove(contents, session=rembg_session)
        
#         # Immediate cleanup
#         del contents
#         gc.collect()
        
#         return StreamingResponse(
#             io.BytesIO(output_data),
#             media_type="image/png",
#             headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"},
#         )
#     except Exception as e:
#         raise HTTPException(500, f"Image processing failed: {e}")

# # Main
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000))
#     print(f"🚀 Starting server on port {port}", flush=True)
    
#     # Pre-warm the model
#     try:
#         logger.info("Warming up model...")
#         # This will download u2netp (11MB) instead of u2net (176MB)
#         gc.collect()
#     except Exception as e:
#         logger.warning(f"Model warm-up failed: {e}")
    
#     uvicorn.run(
#         app, 
#         host="0.0.0.0", 
#         port=port, 
#         log_level="info",
#         timeout_keep_alive=30
#     )

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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