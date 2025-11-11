from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
from typing import Dict

from app.schemas import SummarizeRequest, SummarizeResponse
from app.model import get_model, SummarizationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load model
    logger.info("Starting up... Loading model")
    get_model()  # This loads the model into memory
    logger.info("Model loaded successfully")
    yield
    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="Production-ready API for text summarization using transformers",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information."""
    return {
        "message": "Text Summarization API",
        "version": "1.0.0",
        "endpoints": {
            "POST /summarize": "Summarize text",
            "GET /health": "Health check",
            "GET /docs": "API documentation"
        }
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize_text(
    request: SummarizeRequest,
    model: SummarizationModel = Depends(get_model)
) -> SummarizeResponse:
    """
    Summarize the provided text.
    
    - **text**: The text to summarize (50-10000 characters)
    - **max_length**: Maximum length of summary (30-500 tokens)
    - **min_length**: Minimum length of summary (10-100 tokens)
    - **summary_type**: Preset summary length (short/medium/detailed)
    """
    try:
        # Get length parameters based on summary type
        length_params = model.get_summary_lengths(request.summary_type)
        
        # Override with custom lengths if provided
        max_length = request.max_length or length_params["max_length"]
        min_length = request.min_length or length_params["min_length"]
        
        logger.info(f"Summarizing text of length: {len(request.text)}")
        
        # Generate summary
        summary = model.summarize(
            text=request.text,
            max_length=max_length,
            min_length=min_length
        )
        
        # Calculate metrics
        original_length = len(request.text.split())
        summary_length = len(summary.split())
        compression_ratio = round(summary_length / original_length, 2)
        
        logger.info(f"Summary generated. Compression ratio: {compression_ratio}")
        
        return SummarizeResponse(
            original_length=original_length,
            summary=summary,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )
        
    except Exception as e:
        logger.error(f"Error during summarization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Summarization failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)