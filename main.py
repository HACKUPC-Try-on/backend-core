from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from routers import categories, index, recommendations, image, tryon

app = FastAPI(
    title="Try On backend",
    description='"Try On!" backend for the HackUPC 2024 Inditex Tech challenge',
    docs_url="/docs",
    redoc_url="/redoc",
    version="0.1.0",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1},
)

app.include_router(categories.router)
app.include_router(index.router)
app.include_router(recommendations.router)
app.include_router(image.router)
app.include_router(tryon.router)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")
