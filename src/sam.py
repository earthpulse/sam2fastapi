from fastapi import FastAPI
import numpy as np
from io import BytesIO
from ultralytics import SAM
import rasterio
from rasterio import features
from pyproj import Transformer
import numpy as np

from fastapi import File, UploadFile, Form
from PIL import Image as PILImage
import json
from rasterio.transform import Affine

app = FastAPI()

def process_image_to_geojson(image, points, label, crs, transform, checkpoint_path="sam2_s.pt"):    
    
    crs = rasterio.crs.CRS.from_string(crs)
    transformer_geo_to_crs = Transformer.from_crs("EPSG:4326", crs, always_xy=True)

    model = SAM(checkpoint_path)
    results = model(image, points=points)
    
    combined_mask = np.zeros_like(results[0].masks[0].data.cpu().numpy().squeeze(), dtype=np.uint8)
    for mask in results[0].masks:
        mask_data = mask.data.cpu().numpy().squeeze().astype(np.uint8)
        combined_mask = np.logical_or(combined_mask, mask_data).astype(np.uint8)
    
    shapes = features.shapes(combined_mask, transform=transform)

    polygons = []
    for shape, value in shapes:
        if value != 0:
            coords = shape["coordinates"]
            geo_coords = []
            for ring in coords:
                geo_ring = []
                for point in ring:
                    x, y = point
                    lon, lat = transformer_geo_to_crs.transform(x, y, direction='INVERSE')
                    geo_ring.append([lon, lat])
                geo_coords.append(geo_ring)
            polygons.append(geo_coords)
    
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "MultiPolygon",
            "coordinates": polygons
        },
        "properties": {
            "label": label,
            "task": "segmentation"
        }
    }
    
    return geojson 


@app.post("/sam")
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