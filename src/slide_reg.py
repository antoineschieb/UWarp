import json
from pathlib import Path as P
from time import perf_counter
from PIL import Image, ImageDraw
import numpy as np

from src.core import find_lstsq_solution
from src.data import create_linear_interpolator, find_linear_adjustment, find_nn_adjustment
from src.initial_alignment import inital_registration
from src.linalg import auto_detect_rotation, get_equivalent_region, get_equivalent_region_non_rect
from src.region import Region


class SlideReg:
    def __init__(
        self, name, fixed_slide, moving_slide, N_landmarks, verbose=False
    ) -> None:
        self.name = name
        self.fixed_slide = fixed_slide
        self.moving_slide = moving_slide
        self.N_landmarks = N_landmarks
        self.verbose = verbose
        self.reg_folder = P.cwd() / self.name

        self.rough_transform = None
        self.rotation = None
        self.lstsq_transform = None
        self.input_pts = None
        self.residues = None

    def main_registration(self, downsampling_factor=100, nmi_threshold=0.12) -> None:
        p = self.reg_folder
        p.mkdir(parents=True, exist_ok=True)
        if self.verbose:
            p = self.reg_folder / "all_overlaps"
            p.mkdir(parents=True, exist_ok=True)
            p = self.reg_folder / "final_overlaps"
            p.mkdir(parents=True, exist_ok=True)

        t0 = perf_counter()
        rough_transform = inital_registration(
            self.fixed_slide, self.moving_slide, downsampling_factor=downsampling_factor
        )
        rough_transform = np.linalg.inv(rough_transform)
        rotation = auto_detect_rotation(rough_transform)
        self.rough_transform = rough_transform
        self.rotation = rotation
        if self.verbose:
            print(f"rough_transform: {rough_transform}")
            print(f"rotation: {rotation}")

        self.lstsq_transform, self.input_pts, self.residues, initial_landmarks_nbr = (
            find_lstsq_solution(
                fixed_slide=self.fixed_slide,
                moving_slide=self.moving_slide,
                full_transform=self.rough_transform,
                rotation=self.rotation,
                N_landmarks=self.N_landmarks,
                quality_thresh=nmi_threshold,
                verbose=self.verbose,
                reg_folder=self.reg_folder,
            )
        )

        np.save(P(self.reg_folder / "lstsq_transform.npy"), [self.lstsq_transform])
        np.save(
            P(self.reg_folder / "landmarks_residues.npy"),
            [self.input_pts, self.residues],
        )

        t1 = perf_counter()
        print(f"time: {t1-t0}")

        # Create landmarks img
        img = self.plot_landmarks_on_img()
        img.save(P(self.reg_folder) / "landmarks.jpg")

        # Create summary
        summary = {}
        summary["time"] = t1 - t0
        summary["rough_transform"] = str(self.rough_transform)
        summary["lstsq_transform"] = str(self.lstsq_transform)
        summary["rotation"] = str(self.rotation)
        summary["landmarks_initial"] = str(initial_landmarks_nbr)
        summary["landmarks_kept"] = str(len(self.input_pts))
        summary["avg_residue_length"] = str(
            np.mean(np.linalg.norm(self.residues, axis=1))
        )
        summary["fixed_slide_w_h"] = str(self.fixed_slide.level_dimensions[0])
        with open(P(self.reg_folder / "summary.json"), "w") as f:
            json.dump(summary, f, indent=3)

        return

    def plot_landmarks_on_img(self) -> Image:
        img = self.fixed_slide.get_thumbnail((1024, 1024))
        thmbsize_x, thmbsize_y = img.size
        draw = ImageDraw.Draw(img)

        max_x, max_y = self.fixed_slide.level_dimensions[0]
        for [x, y, _] in self.input_pts:
            pil_x = thmbsize_x * x / max_x
            pil_y = thmbsize_y * y / max_y
            draw.circle([pil_x, pil_y], 6, fill="blue")
        return img


class Warper:
    def __init__(self, reg_folder, fixed_slide, moving_slide):
        self.reg_folder = reg_folder
        self.fixed_slide = fixed_slide
        self.moving_slide = moving_slide

        [self.input_pts, self.residues] = np.load(
            f"{self.reg_folder}/landmarks_residues.npy"
        )
        [self.lstsq_transform] = np.load(f"{self.reg_folder}/lstsq_transform.npy")
        self.interp = create_linear_interpolator(self.input_pts, self.residues)
        self.rotation = auto_detect_rotation(self.lstsq_transform)

    def warp_region(self, r: Region, correction_type=None):
        """
        Warps a region of the fixed slide to a region on the moving slide.
        """
        if correction_type is not None:
            if correction_type == "linear":
                correction = find_linear_adjustment(r.x, r.y, self.interp)
            elif correction_type == "nn":
                correction = find_nn_adjustment(r.x, r.y, self.input_pts, self.residues)
            else:
                raise ValueError("Unknown correction type")
        else:
            correction = np.identity(3)

        r_eq = get_equivalent_region_non_rect(
            r,
            self.fixed_slide,
            self.moving_slide,
            correction @ self.lstsq_transform,
            rotation=self.rotation,
        )
        return r_eq
