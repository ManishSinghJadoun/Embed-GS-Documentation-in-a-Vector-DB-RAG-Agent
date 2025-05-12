from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from app.routes import router  # Assuming your router is in app/routes.py

app = FastAPI()

# Register the router
app.include_router(router)

# Custom error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input format",
            "details": exc.errors()
        },
    )
