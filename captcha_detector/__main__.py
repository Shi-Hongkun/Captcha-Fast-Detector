"""
Command line interface for ROI Calibrator (Step 1).

This CLI supports:
- Calibrate ROI from a directory of training images
- Apply the learned ROI to crop a single image or a directory of images
- Handle paired jpg+txt files
"""

import argparse
import sys
from pathlib import Path

from .roi_calibrator import ROICalibrator


def main():
    """Main entry point for ROI calibration and application."""
    parser = argparse.ArgumentParser(
        description="ROI Calibrator - learn and apply fixed text region"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Calibrate ROI from a directory of images")
    p_cal.add_argument("--input-dir", "-i", required=True, help="Directory with training images")
    p_cal.add_argument("--roi-json", "-r", required=True, help="Path to save ROI JSON")
    p_cal.add_argument("--use-txt", action="store_true", help="Use txt files for calibration instead of images")

    # apply to single image
    p_apply = sub.add_parser("apply", help="Apply ROI to crop a single image")
    p_apply.add_argument("--roi-json", "-r", required=True, help="ROI JSON path")
    p_apply.add_argument("--input", "-i", required=True, help="Input image path")
    p_apply.add_argument("--output", "-o", required=True, help="Output cropped image path")

    # apply to paired files
    p_paired = sub.add_parser("apply-paired", help="Apply ROI to crop paired jpg+txt files")
    p_paired.add_argument("--roi-json", "-r", required=True, help="ROI JSON path")
    p_paired.add_argument("--input-jpg", required=True, help="Input jpg file path")
    p_paired.add_argument("--input-txt", required=True, help="Input txt file path")
    p_paired.add_argument("--output-jpg", required=True, help="Output cropped jpg file path")
    p_paired.add_argument("--output-txt", required=True, help="Output cropped txt file path")

    # batch-apply
    p_batch = sub.add_parser("batch", help="Apply ROI to all images in a directory")
    p_batch.add_argument("--roi-json", "-r", required=True, help="ROI JSON path")
    p_batch.add_argument("--input-dir", "-i", required=True, help="Input directory of images")
    p_batch.add_argument("--output-dir", "-o", required=True, help="Output directory for cropped images")
    p_batch.add_argument("--paired", action="store_true", help="Process paired jpg+txt files")

    args = parser.parse_args()

    try:
        if args.command == "calibrate":
            cal = ROICalibrator()
            if args.use_txt:
                bounds = cal.calibrate_from_txt_dir(args.input_dir)
                print(f"Calibrated ROI from txt files (top, bottom, left, right): {bounds.as_tuple()}")
            else:
                bounds = cal.calibrate_from_dir(args.input_dir)
                print(f"Calibrated ROI from images (top, bottom, left, right): {bounds.as_tuple()}")
            cal.save_bounds(args.roi_json)
            print(f"Saved to: {args.roi_json}")

        elif args.command == "apply":
            cal = ROICalibrator()
            cal.load_bounds(args.roi_json)
            cal.crop_image_file(args.input, args.output)
            print(f"Cropped image saved to: {args.output}")

        elif args.command == "apply-paired":
            cal = ROICalibrator()
            cal.load_bounds(args.roi_json)
            cal.crop_paired_files(args.input_jpg, args.input_txt, args.output_jpg, args.output_txt)
            print(f"Cropped paired files saved to: {args.output_jpg} + {args.output_txt}")

        elif args.command == "batch":
            cal = ROICalibrator()
            cal.load_bounds(args.roi_json)
            
            if args.paired:
                # Process paired jpg+txt files
                cal.crop_paired_dir(args.input_dir, args.output_dir)
                print(f"Cropped paired files (jpg+txt) saved to: {args.output_dir}")
            else:
                # Original behavior: process only images
                in_dir = Path(args.input_dir)
                out_dir = Path(args.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)
                exts = {".jpg", ".jpeg", ".png", ".bmp"}
                for p in sorted(in_dir.iterdir()):
                    if p.suffix.lower() in exts:
                        dst = out_dir / p.name
                        cal.crop_image_file(p, dst)
                print(f"Cropped images saved to: {out_dir}")

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()