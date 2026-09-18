from genie_rover.sdk_client import RoverClient
from PIL import Image
c = RoverClient(timeout=60)
for i in range(1, 6):
    input(f"Poné el rover en la posición {i} y apretá ENTER...")
    img, _ = c.front_frame()
    Image.fromarray(img).save(f"mini_{i}.jpg", quality=92)
    print(f"  guardado mini_{i}.jpg {img.shape}")
