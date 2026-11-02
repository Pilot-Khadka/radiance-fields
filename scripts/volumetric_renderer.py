import numpy as np
import matplotlib.pyplot as plt

from nerf.renderer.scene.scene import Scene
from nerf.renderer.lighting.light import Light
from nerf.renderer.geometry.sphere import Sphere
from nerf.renderer.cameras.pinhole import PinholeCamera
from nerf.renderer.materials.base import VolumeMaterial
from nerf.renderer.renderer.volumetric import VolumetricRenderer


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
    ray = camera.get_ray(0.5, 0.5)
    print("camera pos", camera.position)
    print("ray dir", ray.direction)

    scene = Scene()

    scene.add_object(
        Sphere(
            center=np.array([0, 0.5, 0]),
            radius=0.5,
            material=VolumeMaterial(
                color=np.array([1, 0.3, 0.3]),
                density=5.0,
                absorption=np.array([0.1, 0.5, 0.5]),
                scattering=0.8,
            ),
        )
    ).add_object(
        Sphere(
            center=np.array([-1.2, 0.3, 0.5]),
            radius=0.3,
            material=VolumeMaterial(
                color=np.array([0.3, 1, 0.3]),
                density=6.0,
                absorption=np.array([0.5, 0.1, 0.5]),
                scattering=0.7,
            ),
        )
    ).add_object(
        Sphere(
            center=np.array([1, 0.4, -0.5]),
            radius=0.4,
            material=VolumeMaterial(
                color=np.array([0.3, 0.3, 1]),
                density=4.0,
                absorption=np.array([0.5, 0.5, 0.1]),
                scattering=0.6,
            ),
        )
    )

    scene.add_light(Light(direction=np.array([0.5, 1, 0.3]), color=np.array([1, 1, 1])))

    renderer = VolumetricRenderer(width, height, num_samples=64, step_size=0.05)

    print(f"Rendering {width}x{height} volumetric image...")
    image = renderer.render(scene, camera)

    plt.figure(figsize=(12, 9))
    plt.imshow(image)
    plt.axis("off")
    plt.title("Volumetric Renderer")
    plt.tight_layout()
    plt.show()

    print("Rendering complete!")


if __name__ == "__main__":
    main()
