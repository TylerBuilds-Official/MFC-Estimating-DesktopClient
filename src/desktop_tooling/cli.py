"""
CLI entry point for PDF classification.

Designed for use with desktop UI applications that launch this as a subprocess.

Output Protocol:
- stdout: Final JSON result (single line)
- stderr: Progress messages (newline-delimited JSON)

Usage:
    python -m desktop_tooling <pdf_path> [OPTIONS]
"""
import argparse
import json
import sys
import os


def progress_callback(data: dict) -> None:
    """
    Write progress updates to stderr as JSON.
    Each message is a single line of JSON.
    """
    sys.stderr.write(json.dumps(data) + "\n")
    sys.stderr.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Classify construction drawing PDFs by discipline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m desktop_tooling "C:/Bids/plans.pdf"
    python -m desktop_tooling "C:/Bids/plans.pdf" --output-path "C:/Output"
    python -m desktop_tooling "C:/Bids/plans.pdf" --no-breakout --batch-size 20
        """
    )

    parser.add_argument(
        "pdf_path",
        help="Path to the PDF file to classify"
    )

    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help='Region as JSON dict, e.g. \'{"x_ratio":0.85,"y_ratio":0.92,"w_ratio":0.14,"h_ratio":0.07}\''
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
        "--no-text-first",
        action="store_true",
        help="Disable text-first optimization (use AI for all pages)"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of images per AI API batch (default: 20)"
    )

    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent API batches (default: 8)"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging to pdf_classifier.log"
    )

    args = parser.parse_args()

    # Initialize debug logger
    from .debug_logger import init_logger
    init_logger(enabled=args.debug)

    # Parse region JSON if provided
    region = None
    if args.region:
        try:
            region = json.loads(args.region)
        except json.JSONDecodeError as e:
            error_result = {
                "status": "error",
                "error": f"Invalid region JSON: {e}"
            }
            print(json.dumps(error_result))
            sys.exit(1)

    # Validate PDF path
    if not os.path.exists(args.pdf_path):
        error_result = {
            "status": "error",
            "error": f"PDF not found: {args.pdf_path}"
        }
        print(json.dumps(error_result))
        sys.exit(1)

    try:
        # Import from local classifier module - avoids MCP server initialization
        from .classifier import classify_pdf

        # Call classify_pdf with progress callback
        result = classify_pdf(
            pdf_path=args.pdf_path,
            region=region,
            auto_detect_region=(region is None),
            output_path=args.output_path,
            breakout_files=not args.no_breakout,
            text_first=not args.no_text_first,
            batch_size=args.batch_size,
            max_concurrent=args.max_concurrent,
            progress_callback=progress_callback
        )

        # Output final result as JSON to stdout
        print(json.dumps(result))

        # Exit with appropriate code
        if result.get("status") == "success":
            sys.exit(0)
        else:
            sys.exit(1)

    except Exception as e:
        import traceback
        error_result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
