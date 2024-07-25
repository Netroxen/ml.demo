# ~*~ coding: utf-8 ~*~

from pathlib import Path
from PIL import Image, UnidentifiedImageError

path = Path("v_data/").rglob("*.jpg")
for img_p in path:
    try:
        img = Image.open(img_p)
    except UnidentifiedImageError:
            print(img_p)