from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
from io import BytesIO
from PIL import Image as PILImage
import json
from rasterio.transform import Affine

from src.utils import process_image_to_geojson

app = FastAPI()

@app.get("")
async def root():
    return {"message": "Hello World"}


@app.post("/")
async def sam_endpoint(
    image: UploadFile = File(...),
    points: str = Form(...),
    label: str = Form(...),
    crs: str = Form(...),
    transform: str = Form(...),
):
    try:
        # Read image
        image_data = await image.read()
        img = PILImage.open(BytesIO(image_data))
        rgb = np.array(img)
        # Parse points and transform
        points = json.loads(points)
        transform = Affine.from_gdal(*json.loads(transform))
        # Process to GeoJSON
        geojson = process_image_to_geojson(rgb, points, label, crs, transform)
        return geojson
    except Exception as e:
        return {"error": str(e)}