"""
Pipeline test for the DesktopClient.

Exercises the full flow: ClassificationEngine → BreakoutHandler → AISummaryService.
Mirrors what classifier.py does but with verbose console output for debugging.

Run directly from IDE — edit the constants below.
"""
import os
import sys
import time
from datetime import datetime

# ── CONFIGURE THESE ──────────────────────────────────────────────────────────
INPUT_PDF   = r"C:\Users\tylere.METALSFAB\Desktop\Dev stuff\PDFClassifyMCP\Testing\REV-M_2026-01-19_COMBINED.pdf"
OUTPUT_DIR  = r"D:\Projects\PyCharmProjects\ESTIMATING\PlanClassification\DesktopClient\debug_testing\output"
BREAKOUT    = True
MAX_WORKERS = 8
# ─────────────────────────────────────────────────────────────────────────────

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from plan_classification import (
    ClassificationEngine,
    PipelineConfig,
    PageResult,
    BreakoutHandler,
    AISummaryService,
    DateExtractor,
)

from anthropic import Anthropic


def log(msg: str) -> None:

    """Timestamped console output."""

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")


def run_test():

    """Full pipeline test with verbose output."""

    timestamp = datetime.now().isoformat()
    api_key   = os.getenv('ANTHROPIC_API_KEY')

    log("=" * 70)
    log("DESKTOP CLIENT — FULL PIPELINE TEST")
    log(f"PDF:       {INPUT_PDF}")
    log(f"Output:    {OUTPUT_DIR}")
    log(f"Workers:   {MAX_WORKERS}")
    log(f"Breakout:  {BREAKOUT}")
    log(f"Timestamp: {timestamp}")
    log("=" * 70)

    if not api_key:
        log("ERROR: No ANTHROPIC_API_KEY in .env")
        return

    if not os.path.exists(INPUT_PDF):
        log(f"ERROR: PDF not found: {INPUT_PDF}")
        return

    # ── Phase 1+2: Classification Engine ──────────────────────────────
    log("")
    log("Phase 1+2: Region Detection + Classification")
    log("-" * 70)

    config = PipelineConfig(
        anthropic_api_key=api_key,
        max_workers=MAX_WORKERS,
    )
    engine  = ClassificationEngine(config)
    t_start = time.perf_counter()
    results = engine.classify(INPUT_PDF)
    t_classify = time.perf_counter() - t_start

    log(f"Classified {len(results)} pages in {t_classify:.2f}s")
    log(f"Engine cost: ${engine.total_cost:.4f}")
    log("")

    # ── Results by method ─────────────────────────────────────────────
    method_counts: dict[str, int] = {}
    for r in results:
        method_counts[r.method] = method_counts.get(r.method, 0) + 1

    log("By method:")
    for method, count in sorted(method_counts.items()):
        log(f"  {method:<16} {count:>3} pages")

    # ── Results by discipline ─────────────────────────────────────────
    discipline_counts: dict[str, int] = {}
    for r in results:
        disc = r.discipline or "Unknown"
        discipline_counts[disc] = discipline_counts.get(disc, 0) + 1

    log("By discipline:")
    for disc, count in sorted(discipline_counts.items()):
        log(f"  {disc:<20} {count:>3} pages")

    # ── Per-page detail ───────────────────────────────────────────────
    log("")
    log("-" * 70)
    log("Page details:")
    for r in results:
        log(f"  Page {r.page_index + 1:>3} | {r.method:<12} | {r.sheet_number or 'N/A':<14} | {r.discipline or 'Unknown':<20} | conf={r.confidence:.0%}")

    # ── Engine timings ────────────────────────────────────────────────
    log("")
    log("-" * 70)
    log("Engine timings:")
    for key, val in engine.timings.items():
        log(f"  {key:<24} {val:.2f}s")
    log(f"  {'TOTAL':<24} {t_classify:.2f}s")

    # ── Phase 3: AI Directory Naming ──────────────────────────────────
    output_path = OUTPUT_DIR
    ai_dirname  = None

    if BREAKOUT and output_path:
        log("")
        log("-" * 70)
        log("Phase 3: AI Directory Naming")

        try:
            ai_client       = Anthropic(api_key=api_key)
            summary_service = AISummaryService(client=ai_client)
            pdf_filename    = os.path.basename(INPUT_PDF)
            dir_result      = summary_service.create_dirname(pdf_filename)
            ai_dirname      = dir_result.dir_name.strip()
            ai_dirname      = "".join(c for c in ai_dirname if c not in r'<>:"/\|?*').strip('. ')

            if ai_dirname:
                output_path = os.path.join(output_path, ai_dirname)
                os.makedirs(output_path, exist_ok=True)
                log(f"  AI dirname: {ai_dirname}")
                log(f"  Output:     {output_path}")

        except Exception as e:
            log(f"  AI dirname failed (non-fatal): {e}")

    # ── Phase 4: Date Extraction (tiered: text → OCR → vision) ───────
    date_map = None

    if BREAKOUT:
        log("")
        log("-" * 70)
        log("Phase 4: Date Extraction (tiered)")

        try:
            ai_client = Anthropic(api_key=api_key)
            extractor = DateExtractor(anthropic_client=ai_client)
            date_map  = extractor.extract_all(INPUT_PDF, results)

            for disc, date_val in date_map.items():
                log(f"  {disc:<20} -> {date_val}")

        except Exception as e:
            log(f"  Date extraction failed (non-fatal): {e}")
            date_map = None

    # ── Phase 5: Breakout Files ───────────────────────────────────────
    if BREAKOUT:
        log("")
        log("-" * 70)
        log("Phase 5: Breakout Files")

        results_dicts = [r.to_dict() for r in results]

        handler = BreakoutHandler(
            classification_results=results_dicts,
            pdf_path=INPUT_PDF,
            output_dir=output_path,
        )
        breakout_result = handler.breakout(date_map=date_map)

        for f in breakout_result.get('created_files', []):
            disc = f.get('discipline', f.get('category', 'Unknown'))
            log(f"  {disc:<20} {f['page_count']:>3} pages -> {os.path.basename(f['output_path'])}")

    # ── Done ──────────────────────────────────────────────────────────
    log("")
    log("=" * 70)
    log(f"TOTAL TIME:  {t_classify:.2f}s (classification only)")
    log(f"TOTAL COST:  ${engine.total_cost:.4f}")
    log("DONE")
    log("=" * 70)


if __name__ == "__main__":
    run_test()
