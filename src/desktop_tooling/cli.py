"""
CLI entry point for PDF classification.

Designed for use with desktop UI applications that launch this as a subprocess.

Output Protocol:
    stdout: Final JSON result (single line)
    stderr: Progress messages (newline-delimited JSON)
"""
import argparse
import json
import sys
import os


def progress_callback(data: dict) -> None:

    """Write progress updates to stderr as JSON."""

    sys.stderr.write(json.dumps(data) + "\n")
    sys.stderr.flush()


def main():

    """Parse args, validate inputs, run classification pipeline."""

    parser = argparse.ArgumentParser(
        description="Classify construction drawing PDFs by discipline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m desktop_tooling "C:/Bids/plans.pdf"
    python -m desktop_tooling "C:/Bids/plans.pdf" --output-path "C:/Output"
    python -m desktop_tooling "C:/Bids/plans.pdf" --no-breakout --max-workers 10
        """
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to classify"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output directory for breakout files (default: ~/Desktop/breakout)"
    )

    parser.add_argument(
        "--no-breakout",
        action="store_true",
        help="Skip creating breakout files"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Max concurrent API workers (default: 8)"
    )

    parser.add_argument(
        "--breakout-filter",
        type=str,
        choices=["all", "standard"],
        default="all",
        help="Filter breakout files: 'all' for every discipline, 'standard' for Arch/Struct/Land/Civil only (default: all)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to ~/pdf_classifier.log"
    )

    args = parser.parse_args()

    # Initialize debug logger
    from .debug_logger import init_logger
    init_logger(enabled=args.debug)

    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        error_result = {
            "status": "error",
            "error": f"PDF not found: {args.pdf_path}",
        }
        print(json.dumps(error_result))
        sys.exit(1)

    try:
        from .classifier import classify_pdf

        result = classify_pdf(
            pdf_path=args.pdf_path,
            output_path=args.output_path,
            breakout_files=not args.no_breakout,
            breakout_filter=args.breakout_filter,
            max_workers=args.max_workers,
            progress_callback=progress_callback,
        )

        print(json.dumps(result))
        sys.exit(0 if result.get("status") == "success" else 1)

    except Exception as e:
        import traceback
        error_result = {
            "status":    "error",
            "error":     str(e),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
