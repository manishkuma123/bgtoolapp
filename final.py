import os, io
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import httpx

app = FastAPI(title="Background Removal API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API key from environment variable
REMOVEBG_API_KEY = os.environ.get("REMOVEBG_API_KEY", "fyuP441HEmhgwjbrgEVPeNWJ") 

@app.get("/")
def root():
    return {"status": "running", "service": "remove.bg API"}

@app.post("/remove-image-background/")
async def remove_bg(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Please upload an image")
    
    if not REMOVEBG_API_KEY:
        raise HTTPException(500, "API key not configured")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:  # 10MB limit
        raise HTTPException(400, "Max 10MB image")

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                "https://api.remove.bg/v1.0/removebg",
                files={"image_file": (file.filename, contents, file.content_type)},
                data={"size": "auto"},
                headers={"X-Api-Key": REMOVEBG_API_KEY}
            )
            
            if response.status_code != 200:
                error_msg = response.json().get("errors", [{}])[0].get("title", "Unknown error")
                raise HTTPException(response.status_code, f"Remove.bg error: {error_msg}")
            
            return StreamingResponse(
                io.BytesIO(response.content),
                media_type="image/png",
                headers={
                    "Content-Disposition": f"attachment; filename=no_bg_{file.filename}",
                    "Access-Control-Expose-Headers": "Content-Disposition"
                }
            )
        except httpx.TimeoutException:
            raise HTTPException(504, "Request timeout - image processing took too long")
        except httpx.RequestError as e:
            raise HTTPException(500, f"Network error: {str(e)}")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/health")
def health():
    return {"status": "ok", "api_configured": bool(REMOVEBG_API_KEY)}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)