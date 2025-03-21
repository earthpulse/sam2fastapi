from ultralytics import SAM
import rasterio
from rasterio import features
from pyproj import Transformer
import numpy as np

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
