# 🔀 UWarp: Landmark-Based WSI Coordinate Warping

**UWarp** is a registration tool tailored for aligning histopathology Whole Slide Images (WSIs) scanned under different conditions (e.g., different scanners or acquisition settings). It enables **precise patch-level alignment** between slides, a critical requirement for analyzing **scanner-induced domain shift** in computational pathology.

📄 **Read the paper:**  
[UWarp: A Whole Slide Image Registration Pipeline to Characterize Scanner-Induced Local Domain Shift](https://arxiv.org/abs/2503.20653)

---

## 🧠 Why UWarp?

Slide digitization introduces **domain shifts** that affect the reliability of deep learning models in pathology. While existing methods quantify this shift at the dataset or slide level, **UWarp** enables **localized (patch-level)** alignment and comparison.

UWarp is particularly suited for:

- Scanner-to-scanner slide alignment  
- Visual QA of patch-level correspondence  
- Domain shift characterization and TRE evaluation
- Robust multi-scanner model evaluation

It combines:

- ✅ **Global affine** registration  
- ✅ **Local correction** via polynomial or linear warping  
- ✅ Fast, interpretable, reproducible warps

---

## 📦 Installation

Install dependencies using `poetry` and the provided `pyproject.toml`:

```bash
poetry install
```

## 🧪 Example Usage

```python
# Step 1: Register two slides, and save the transformation parameters
r = SlideReg(reg_id, fixed_slide, moving_slide, N_landmarks=50, verbose=False)
r.main_registration()
```

- `fixed_slide` and `moving_slide` must expose an OpenSlide-like API (`dimensions`, `read_region`, etc.).
- A folder containing transformation parameters will be created under the provided `reg_id`.

```python
# Step 2: Create a warper from the saved transformation parameters
warper = Warper(reg_id, fixed_slide, moving_slide)

# Step 3: Sample regions and visualize the overlap
coords = get_random_valid_points(find_slidepath_for_scanner_Cypath(sample_id, scanner_fixed),
                                 fixed_slide.dimensions, N=3)

for x, y in coords:
    r = Region(x, y, 0, 512, 512)
    r_eq = warper.warp_region(r, correction_type="linear")
    p0 = get_patch(fixed_slide, r)
    p1 = get_patch(moving_slide, r_eq)

    if np.sum(p0) == 0 or np.sum(p1) == 0:
        continue

    display(display_overlapping_patches(p0, p1, rotate=warper.rotation))
```

---

## 📊 Performance & Validation

In our experiments across two private multi-scanner datasets (CypathLung and BosomShieldBreast), **UWarp achieved a median target registration error (TRE) of < 4 pixels** at 40× magnification (~1 μm), significantly outperforming state-of-the-art open-source methods (~5 μm).

UWarp is also used to analyze **localized prediction variability** in deep learning models, showing correlations with **tissue content** and **nuclei density**, and helping identify areas vulnerable to domain shift.

---

## 📝 Notes

- Supports multiple interpolation modes (e.g., `"linear"`, `"nearest_neighbor"`).  
- A custom NMI threshold can be set for more or less permissive landmark quality control. 
- Light dependencies, easy to plug into downstream patch-level pipelines.

---
