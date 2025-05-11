from fastapi import FastAPI
from .routes import router

# Initialize the FastAPI app
app = FastAPI()

# Include the router to handle the routes
app.include_router(router)
