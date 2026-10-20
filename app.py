import base64
import io

import matplotlib

# Set non-interactive backend before importing pyplot (required for web servers)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, render_template, request
from pystac_client import Client
from odc.stac import configure_rio, load

app = Flask(__name__)

# Configure rasterio for anonymous AWS S3 cloud access
configure_rio(cloud_defaults=True, aws={"aws_unsigned": True})

# Earth Search STAC API endpoint
STAC_API_URL = "https://earth-search.aws.element84.com/v1"


@app.route("/")
def index():
    # Default initial coordinates (Chennai)
    return render_template("index.html", lat=13.08, lon=80.27)


@app.route("/get_map_data", methods=["POST"])
def get_map_data():
    try:
        req = request.json
        lat = float(req["lat"])
        lon = float(req["lon"])
        year = int(req["year"])
        layer = req.get("layer", "ndvi")  # Options: 'ndvi' or 'rgb'

        # Create ~4km x 4km bounding box around the target coordinates
        delta = 0.02
        bbox = [lon - delta, lat - delta, lon + delta, lat + delta]
        date_range = f"{year}-01-01/{year}-12-31"

        # Search Element 84 STAC catalog for cloud-free Sentinel-2 images
        client = Client.open(STAC_API_URL)
        search = client.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": 5}},
            limit=1,
        )

        items = list(search.items())
        if not items:
            return jsonify(
                {
                    "error": f"No cloud-free imagery (<5% cover) found for {year} at these coordinates."
                }
            )

        item = items[0]
        capture_date = item.properties["datetime"][:10]

        # Initialize plot
        fig, ax = plt.subplots(figsize=(6, 6), dpi=100)

        if layer == "ndvi":
            # Lazy-load Red (B4) and NIR (B8)
            ds = load([item], bands=["red", "nir"], bbox=bbox, resolution=10)
            red = ds["red"].isel(time=0).values.astype(np.float32)
            nir = ds["nir"].isel(time=0).values.astype(np.float32)

            # Compute NDVI: (NIR - Red) / (NIR + Red)
            np.seterr(divide="ignore", invalid="ignore")
            ndvi = (nir - red) / (nir + red)
            ndvi = np.nan_to_num(ndvi)

            # Render with optimized limits for high urban/vegetation contrast
            im = ax.imshow(ndvi, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
            fig.colorbar(im, ax=ax, label="NDVI Value", fraction=0.046, pad=0.04)
            ax.set_title(
                f"NDVI Vegetation Index ({capture_date})", fontweight="bold"
            )

        else:  # 'rgb' True Color
            # Lazy-load Red (B4), Green (B3), Blue (B2)
            ds = load(
                [item], bands=["red", "green", "blue"], bbox=bbox, resolution=10
            )
            r = ds["red"].isel(time=0).values.astype(np.float32)
            g = ds["green"].isel(time=0).values.astype(np.float32)
            b = ds["blue"].isel(time=0).values.astype(np.float32)

            # Normalize 0-10000 reflectance scale to 0-1 range (3000 brightening scalar)
            rgb = np.stack([r, g, b], axis=-1)
            rgb = np.clip(rgb / 3000.0, 0, 1)

            ax.imshow(rgb)
            ax.set_title(
                f"True Color Satellite Photo ({capture_date})",
                fontweight="bold",
            )

        ax.axis("off")
        fig.tight_layout()

        # Save plot to in-memory PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # Explicitly close figures to prevent memory leaks in RAM
        plt.close(fig)
        plt.close("all")

        return jsonify(
            {"date": capture_date, "image": f"data:image/png;base64,{img_base64}"}
        )

    except Exception as e:
        plt.close("all")
        return jsonify({"error": f"Server processing error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)