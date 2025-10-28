import os, io, gc
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove, new_session

# ---- Memory limiting ----
os.environ.update({
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "ONNX_DISABLE_EXTERNAL_CUSTOM_OPS": "1",
    "ORT_DISABLE_ALL_OPTIMIZATIONS": "1",
    "U2NET_HOME": "/tmp/.u2net",
})

app = FastAPI(title="Render 512MB Image Remover")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Lazy init
rembg_session = None

@app.get("/")
def root():
    return {"status": "running", "model": "u2netp"}

@app.post("/remove-image-background/")
async def remove_bg(file: UploadFile = File(...)):
    global rembg_session
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image")

    if rembg_session is None:
        rembg_session = new_session(model_name="u2netp")

    contents = await file.read()
    if len(contents) > 5 * 1024 * 1024:
        raise HTTPException(400, "Max 5MB image")

    output = remove(contents, session=rembg_session)
    del contents
    gc.collect()

    return StreamingResponse(
        io.BytesIO(output),
        media_type="image/png",
        headers={"Content-Disposition": f"attachment; filename=no_bg_{file.filename}"}
    )

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn, sys, logging
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
