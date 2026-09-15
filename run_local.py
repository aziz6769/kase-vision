from dotenv import load_dotenv
from pathlib import Path
import os
import uvicorn

BASE = Path(__file__).resolve().parent
load_dotenv(BASE / ".env")

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
