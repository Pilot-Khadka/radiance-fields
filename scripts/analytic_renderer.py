import numpy as np
import matplotlib.pyplot as plt

from nerf.renderer.scene.scene import Scene
from nerf.renderer.lighting.light import Light
from nerf.renderer.geometry.plane import Plane
from nerf.renderer.geometry.sphere import Sphere
from nerf.renderer.cameras.pinhole import PinholeCamera
from nerf.renderer.materials.phong import PhongMaterial
from nerf.renderer.renderer.raytracer import RayTracerRenderer


def main():
    width, height = 800, 600

    camera = PinholeCamera(
        position=np.array([4.0, 2.0, 4.0]),
        look_at=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        fov=60,
        width=width,
        height=height,
    )

    scene = Scene()

    scene.add_object(
        Sphere(
            np.array([0, 0.5, 0]),
            0.5,
            PhongMaterial(np.array([1, 0.3, 0.3]), specular=0.8),
        )
    ).add_object(
        Sphere(
            np.array([-1.2, 0.3, 0.5]),
            0.3,
            PhongMaterial(np.array([0.3, 1, 0.3]), specular=0.6),
        )
    ).add_object(
        Sphere(
            np.array([1, 0.4, -0.5]),
            0.4,
            PhongMaterial(np.array([0.3, 0.3, 1]), specular=0.4),
        )
    ).add_object(
        Plane(
            np.array([0, 0, 0]),
            np.array([0, 1, 0]),
            PhongMaterial(np.array([0.8, 0.8, 0.8]), specular=0.1),
        )
    )

    scene.add_light(Light(direction=np.array([0.5, 1, 0.3]), color=np.array([1, 1, 1])))

    renderer = RayTracerRenderer(width, height)

    print(f"Rendering {width}x{height} image...")
    image = renderer.render(scene, camera)

    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Analytic Renderer")
    plt.tight_layout()
    plt.show()

    print("Rendering complete!")


if __name__ == "__main__":
    main()
