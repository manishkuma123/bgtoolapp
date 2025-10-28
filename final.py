import os
import gc
import io
import sys
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session

# --- Ultra low memory settings ---
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["ONNX_DISABLE_EXTERNAL_CUSTOM_OPS"] = "1"
os.environ["U2NET_HOME"] = "/tmp/.u2net"

# --- Logging ---
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger("LightweightRembg")

# --- FastAPI setup ---
app = FastAPI(title="Lightweight Image Background Remover")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Initialize lightweight model ---
logger.info("⚙️ Loading u2netp model (11 MB)...")
rembg_session = new_session(model_name="u2netp")

@app.get("/")
async def root():
    return {
        "status": "running",
        "model": "u2netp (11MB)",
        "note": "Image-only background remover for Render 512 MB tier"
    }

@app.post("/remove-image-background/")
async def remove_image_background(file: UploadFile = File(...)):
    """Remove background from a single image (max 5 MB)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image file")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(400, "Image too large (max 5MB)")

    try:
        output_data = remove(contents, session=rembg_session)
        del contents
        gc.collect()
        return StreamingResponse(
            io.BytesIO(output_data),
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=no_bg_{file.filename}"
            },
        )
    except Exception as e:
        logger.error(f"❌ Image processing failed: {e}")
        raise HTTPException(500, f"Processing failed: {e}")

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "u2netp"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    logger.info(f"🚀 Starting image remover on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
