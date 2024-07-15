import os
import platform
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Optional, Union

import blobfile as bf
import numpy as np
from PIL import Image

from ply_util import write_ply


UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]
BASIC_AMBIENT_COLOR = 0.3
BASIC_DIFFUSE_COLOR = 0.7


@dataclass
class TriMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: np.ndarray

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: np.ndarray

    # [P x 3] array of normal vectors per face.
    normals: Optional[np.ndarray] = None

    # Extra data per vertex and face.
    vertex_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "TriMesh":
        """
        Load the mesh from a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            verts = obj["verts"]
            faces = obj["faces"]
            normals = obj["normals"] if "normals" in keys else None
            vertex_channels = {}
            face_channels = {}
            for key in keys:
                if key.startswith("v_"):
                    vertex_channels[key[2:]] = obj[key]
                elif key.startswith("f_"):
                    face_channels[key[2:]] = obj[key]
            return cls(
                verts=verts,
                faces=faces,
                normals=normals,
                vertex_channels=vertex_channels,
                face_channels=face_channels,
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the mesh to a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "wb") as writer:
                self.save(writer)
        else:
            obj_dict = dict(verts=self.verts, faces=self.faces)
            if self.normals is not None:
                obj_dict["normals"] = self.normals
            for k, v in self.vertex_channels.items():
                obj_dict[f"v_{k}"] = v
            for k, v in self.face_channels.items():
                obj_dict[f"f_{k}"] = v
            np.savez(f, **obj_dict)

    def has_vertex_colors(self) -> bool:
        return self.vertex_channels is not None and all(x in self.vertex_channels for x in "RGB")

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.verts,
            rgb=(
                np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
                if self.has_vertex_colors()
                else None
            ),
            faces=self.faces,
        )

    def write_obj(self, raw_f: BinaryIO):
        if self.has_vertex_colors():
            vertex_colors = np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
            vertices = [
                "{} {} {} {} {} {}".format(*coord, *color)
                for coord, color in zip(self.verts.tolist(), vertex_colors.tolist())
            ]
        else:
            vertices = ["{} {} {}".format(*coord) for coord in self.verts.tolist()]

        faces = [
            "f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1))
            for tri in self.faces.tolist()
        ]

        combined_data = ["v " + vertex for vertex in vertices] + faces

        raw_f.writelines("\n".join(combined_data))


SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blender_script.py")


def render_model(
    model_path: str,
    output_path: str,
    num_images: int,
    backend: str = "BLENDER_EEVEE",
    light_mode: str = "random",
    camera_pose: str = "random",
    camera_dist_min: float = 2.0,
    camera_dist_max: float = 2.0,
    fast_mode: bool = False,
    extract_material: bool = False,
    delete_material: bool = False,
    verbose: bool = False,
    timeout: float = 60 * 60,
):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_in = model_path
        tmp_out = os.path.join(tmp_dir, "out")
        zip_out = tmp_out + ".zip"
        os.mkdir(tmp_out)
        args = []
        if platform.system() == "Linux":
            # Needed to enable Eevee backend on headless linux.
            args = ["xvfb-run", "-a"]
        args.extend(
            [
                _blender_binary_path(),
                "-b",
                "-P",
                SCRIPT_PATH,
                "--",
                "--input_path",
                tmp_in,
                "--output_path",
                tmp_out,
                "--num_images",
                str(num_images),
                "--backend",
                backend,
                "--light_mode",
                light_mode,
                "--camera_pose",
                camera_pose,
                "--camera_dist_min",
                str(camera_dist_min),
                "--camera_dist_max",
                str(camera_dist_max),
                "--uniform_light_direction",
                *[str(x) for x in UNIFORM_LIGHT_DIRECTION],
                "--basic_ambient",
                str(BASIC_AMBIENT_COLOR),
                "--basic_diffuse",
                str(BASIC_DIFFUSE_COLOR),
            ]
        )
        if fast_mode:
            args.append("--fast_mode")
        if extract_material:
            args.append("--extract_material")
        if delete_material:
            args.append("--delete_material")
        if verbose:
            subprocess.check_call(args)
        else:
            try:
                output = subprocess.check_output(args, stderr=subprocess.STDOUT, timeout=timeout)
            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"{exc}: {exc.output}") from exc
        if not os.path.exists(os.path.join(tmp_out, "info.json")):
            if verbose:
                # There is no output available, since it was
                # logged directly to stdout/stderr.
                raise RuntimeError(f"render failed: output file missing")
            else:
                raise RuntimeError(f"render failed: output file missing. Output: {output}")
        _combine_rgba(tmp_out)
        with zipfile.ZipFile(zip_out, mode="w") as zf:
            for name in os.listdir(tmp_out):
                zf.write(os.path.join(tmp_out, name), name)
        bf.copy(zip_out, output_path, overwrite=True)


def render_mesh(
    mesh: TriMesh,
    output_path: str,
    num_images: int,
    backend: str = "BLENDER_EEVEE",
    camera_pose: str = "random", # z-circular
    **kwargs,
):
    if mesh.has_vertex_colors() and backend not in ["BLENDER_EEVEE", "CYCLES"]:
        raise ValueError(f"backend does not support vertex colors: {backend}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        ply_path = os.path.join(tmp_dir, "out.ply")
        with open(ply_path, "wb") as f:
            mesh.write_ply(f)
        render_model(
            ply_path, output_path=output_path, num_images=num_images,
            backend=backend, camera_pose=camera_pose, **kwargs
        )


def _combine_rgba(out_dir: str):
    i = 0
    while True:
        paths = [os.path.join(out_dir, f"{i:05}_{ch}.png") for ch in "rgba"]
        if not os.path.exists(paths[0]):
            break
        joined = np.stack(
            [(np.array(Image.open(path)) >> 8).astype(np.uint8) for path in paths], axis=-1
        )
        Image.fromarray(joined).save(os.path.join(out_dir, f"{i:05}.png"))
        for path in paths:
            os.remove(path)
        i += 1


def _blender_binary_path() -> str:
    path = os.getenv("BLENDER_PATH", None)
    if path is not None:
        return path

    if os.path.exists("/Applications/Blender.app/Contents/MacOS/Blender"):
        return "/Applications/Blender.app/Contents/MacOS/Blender"

    raise EnvironmentError(
        "To render 3D models, install Blender version 3.3.1 or higher and "
        "set the environment variable `BLENDER_PATH` to the path of the Blender executable."
    )