import os
import numpy as np
import pandas as pd
import pydicom
from pathlib import Path
import SimpleITK as sitk
from radiomics import featureextractor
from rt_utils import RTStructBuilder
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)  # suppress radiomics warnings

# ----------------------------
# Helper: Load CT as SimpleITK image
# ----------------------------
def load_ct_as_sitk(ct_folder):
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(ct_folder))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {ct_folder}")
    reader.SetFileNames(files)
    return reader.Execute()

# ----------------------------
# Helper: Extract from DICOM-SEG
# ----------------------------
def extract_tumor_mask_from_seg(seg_file, ct_img, ct_folder):
    seg_ds = pydicom.dcmread(str(seg_file), force=True)
    dcm_files = [f for f in Path(ct_folder).rglob("*.dcm") if f.is_file()]
    ct_slices = [pydicom.dcmread(str(f), force=True) for f in dcm_files]
    ct_slices.sort(key=lambda s: float(getattr(s, 'ImagePositionPatient', [0,0,0])[2]))
    ct_sop_to_idx = {s.SOPInstanceUID: i for i, s in enumerate(ct_slices)}

    H, W = seg_ds.Rows, seg_ds.Columns
    tumor_mask_3d = np.zeros((len(ct_slices), H, W), dtype=np.uint8)

    tumor_seg_idx = None
    tumor_keywords = ["Neoplasm, Primary", "GTV", "GTV-1", "Tumor"]
    for i, seg in enumerate(seg_ds.SegmentSequence):
        label = getattr(seg, 'SegmentLabel', f"Segment-{getattr(seg, 'SegmentNumber', i+1)}")
        if any(kw in str(label) for kw in tumor_keywords):
            tumor_seg_idx = i + 1
            break

    if tumor_seg_idx is None:
        raise ValueError("Tumor label not found")

    for frame_idx in range(getattr(seg_ds, 'NumberOfFrames', 0)):
        try:
            seg_num = seg_ds.PerFrameFunctionalGroupsSequence[frame_idx].SegmentIdentificationSequence[0].ReferencedSegmentNumber
            if seg_num != tumor_seg_idx:
                continue
            ref_sop = seg_ds.PerFrameFunctionalGroupsSequence[frame_idx].DerivationImageSequence[0].SourceImageSequence[0].ReferencedSOPInstanceUID
            if ref_sop in ct_sop_to_idx:
                slice_idx = ct_sop_to_idx[ref_sop]
                frame_data = seg_ds.pixel_array[frame_idx]
                tumor_mask_3d[slice_idx] = frame_data
        except (IndexError, AttributeError, KeyError):
            continue

    mask_sitk = sitk.GetImageFromArray(tumor_mask_3d)
    mask_sitk.CopyInformation(ct_img)
    return mask_sitk

# ----------------------------
# Helper: Extract from RTSTRUCT
# ----------------------------
def extract_tumor_mask_from_rtstruct(rtstruct_file, ct_folder):
    rtstruct = RTStructBuilder.create_from(
        dicom_series_path=str(ct_folder),
        rt_struct_path=str(rtstruct_file)
    )
    roi_names = rtstruct.get_roi_names()
    target_roi = None
    for name in roi_names:
        if any(kw in name for kw in ["GTV-1", "GTV", "Tumor", "Primary"]):
            target_roi = name
            break
    if not target_roi:
        target_roi = roi_names[0]  # fallback to first ROI

    mask_3d = rtstruct.get_roi_mask_by_name(target_roi)
    mask_array = np.transpose(mask_3d.astype(np.uint8), (2, 0, 1))  # (D, H, W)

    ct_img = load_ct_as_sitk(ct_folder)
    mask_sitk = sitk.GetImageFromArray(mask_array)
    mask_sitk.CopyInformation(ct_img)
    return mask_sitk

# ----------------------------
# Main Loop: All 422 Patients
# ----------------------------
csv_path = "/content/ct_seg_paths.csv"
df = pd.read_csv(csv_path)

# Handle "NA" strings: replace with NaN, then check with pd.isna
df = df.replace("NA", np.nan)

extractor = featureextractor.RadiomicsFeatureExtractor()
results = []
total_patients = len(df)
print(f"Processing {total_patients} patients...")

for i, row in df.iterrows():
    patient_id = row["patient_id"]
    ct_folder = Path(row["ct_folder_path"])
    seg_file = row["seg_file_path"]
    rtstruct_file = Path(row["rat_file_path"])

    if i % 50 == 0:
        print(f"\nProgress: {i}/{total_patients} patients...")

    features = {"patient_id": patient_id}

    try:
        ct_img = load_ct_as_sitk(ct_folder)

        # --- SEG PATH: only if valid ---
        seg_valid = False
        if pd.notna(seg_file):
            seg_path = Path(seg_file)
            if seg_path.is_file():
                try:
                    mask_seg = extract_tumor_mask_from_seg(seg_path, ct_img, ct_folder)
                    feat = extractor.execute(ct_img, mask_seg)
                    features.update({k + "_seg": v for k, v in feat.items() if not k.startswith("diagnostics_")})
                    seg_valid = True
                except Exception as e:
                    print(f"  ️ {patient_id} SEG failed: {str(e)[:100]}")
            else:
                print(f"     {patient_id}: SEG file not found")
        if not seg_valid:
            # Optionally log that SEG was skipped
            pass

        # --- RTSTRUCT: always attempt ---
        try:
            mask_rt = extract_tumor_mask_from_rtstruct(rtstruct_file, ct_folder)
            feat = extractor.execute(ct_img, mask_rt)
            features.update({k + "_rtstruct": v for k, v in feat.items() if not k.startswith("diagnostics_")})
        except Exception as e:
            print(f"     {patient_id} RTSTRUCT failed: {str(e)[:100]}")
            # Even if RTSTRUCT fails, we might still have SEG → still append
            if not any(k.endswith('_seg') for k in features.keys() if k != "patient_id"):
                continue  # skip if both failed

        results.append(features)

    except Exception as e:
        print(f"    {patient_id} overall error: {str(e)[:100]}")
        continue

# ----------------------------
# Final Save
# ----------------------------
if not results:
    raise ValueError("No patients succeeded!")

feature_df = pd.DataFrame(results)
feature_df.set_index("patient_id", inplace=True)

output_path = "radiomics_features_422_patients_seg_rtstruct.csv"
feature_df.to_csv(output_path)
print(f"\nDone! Saved features for {len(feature_df)} patients to '{output_path}'.")

# Summary
seg_cols = [c for c in feature_df.columns if c.endswith('_seg')]
rt_cols = [c for c in feature_df.columns if c.endswith('_rtstruct')]
print(f"\nFeature counts:")
print(f"- Patients with SEG features:   {feature_df[seg_cols].notna().any(axis=1).sum() if seg_cols else 0}")
print(f"- Patients with RTSTRUCT features: {feature_df[rt_cols].notna().any(axis=1).sum() if rt_cols else 0}")
print(f"- Total radiomics features (SEG + RTSTRUCT): {len(feature_df.columns) - 1}")