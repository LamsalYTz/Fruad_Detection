from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("API_KEY", "fruad-secret-key-2026")
API_HEADER = APIKeyHeader(name="X-API-KEY", auto_error=False)

def verify_security_key(key : str = Security(API_HEADER)):
    if key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or Missing Key!"
        )
    
    return key