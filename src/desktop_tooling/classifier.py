"""
Self-contained PDF classification logic with progress callbacks.

Thin orchestration layer: builds PipelineConfig, calls ClassificationEngine,
then runs breakout/summary phases for desktop UI integration.

Imports only from plan_classification (NOT MCP server tools).
"""
import os
import sys
from typing import Dict, List, Callable

from anthropic import Anthropic
from dotenv import load_dotenv

# In production (frozen), API keys are injected by the parent process.
# In development, load from .env for local testing.
if not getattr(sys, 'frozen', False):
    load_dotenv()

from . import debug_logger as log

from plan_classification import (
    ClassificationEngine,
    PipelineConfig,
    PageResult,
    RegionDetectionError,
    BreakoutHandler,
    AISummaryService,
    DateExtractor,
)


# Disciplines included in "standard" breakout mode
STANDARD_DISCIPLINES = {'Architectural', 'Structural', 'Landscape', 'Civil', 'Unknown'}


def classify_pdf(
        pdf_path: str,
        output_path: str | None = None,
        breakout_files: bool = True,
        breakout_filter: str = 'all',
        max_workers: int = 8,
        progress_callback: Callable[[Dict], None] | None = None ) -> Dict:

    """Classify pages in a construction drawing PDF by discipline."""

    anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
    if not anthropic_api_key:

        return {"status": "error", "error": "No ANTHROPIC_API_KEY found in .env"}

    if not os.path.exists(pdf_path):

        return {"status": "error", "error": f"PDF not found: {pdf_path}"}

    def _report(data: Dict) -> None:

        """Write progress update to log and optional callback."""

        log.log_phase(data.get('phase', ''), data.get('status', ''), **{
            k: v for k, v in data.items() if k not in ('type', 'phase', 'status')
        })
        if progress_callback:
            progress_callback(data)

    try:
        # ── Phase 1+2: Region detection + Classification ──────────────
        def _engine_progress(data: Dict) -> None:

            """Bridge engine callbacks into desktop progress stream."""

            _report({"type": "progress", **data})

        config = PipelineConfig(
            anthropic_api_key=anthropic_api_key,
            max_workers=max_workers,
        )
        engine  = ClassificationEngine(config)
        results = engine.classify(pdf_path, logger=log, on_progress=_engine_progress)

        results_dicts = [r.to_dict() for r in results]

        # ── AI Client (shared for dirname + summary + dates) ──────────
        ai_client = Anthropic(api_key=anthropic_api_key)

        # ── Phase 3: AI Directory Naming ──────────────────────────────
        ai_dirname = None
        if output_path:
            try:
                _report({"type": "progress", "phase": "ai_dirname", "status": "started"})

                summary_service = AISummaryService(client=ai_client)
                pdf_filename    = os.path.basename(pdf_path)
                dir_result      = summary_service.create_dirname(pdf_filename)
                ai_dirname      = dir_result.dir_name.strip()
                ai_dirname      = "".join(c for c in ai_dirname if c not in r'<>:"/\|?*').strip('. ')

                if ai_dirname:
                    output_path = os.path.join(output_path, ai_dirname)
                    os.makedirs(output_path, exist_ok=True)

                _report({"type": "progress", "phase": "ai_dirname", "status": "completed", "dirname": ai_dirname})

            except Exception as e:
                log.error(f'AI dirname failed: {e}', exc=e)
                _report({"type": "progress", "phase": "ai_dirname", "status": "failed", "error": str(e)})

        # ── Phase 4: Date Extraction (tiered: text → OCR → vision) ────
        date_map = None
        if breakout_files:
            try:
                _report({"type": "progress", "phase": "date_extraction", "status": "started"})

                extractor = DateExtractor(anthropic_client=ai_client)
                date_map  = extractor.extract_all(pdf_path, results, logger=log)

                _report({"type": "progress", "phase": "date_extraction", "status": "completed"})

            except Exception as e:
                log.error(f'Date extraction failed: {e}', exc=e)
                _report({"type": "progress", "phase": "date_extraction", "status": "failed", "error": str(e)})
                date_map = None

        # ── Phase 5: Breakout Files ───────────────────────────────────
        created_files = {}
        if breakout_files:
            _report({"type": "progress", "phase": "breakout", "status": "creating_files"})

            # Always merge General → Architectural for breakout
            breakout_results = [
                {**r, 'discipline': 'Architectural'}
                if (r.get('discipline') or r.get('category', 'Unknown')) == 'General'
                else r
                for r in results_dicts
            ]

            # Filter results for breakout if using standard mode
            if breakout_filter == 'standard':
                breakout_results = [
                    r for r in breakout_results
                    if (r.get('discipline') or r.get('category', 'Unknown')) in STANDARD_DISCIPLINES
                ]

            def _breakout_progress(current, total, discipline_name):

                _report({
                    "type": "progress",
                    "phase": "breakout",
                    "status": "creating_files",
                    "current": current,
                    "total": total,
                    "category": discipline_name,
                })

            created_files = _run_breakout(
                results=breakout_results,
                pdf_path=pdf_path,
                output_path=output_path,
                on_progress=_breakout_progress,
                date_map=date_map,
            )

            # Split merged Architectural back into separate display rows
            created_files = _split_general_display(created_files, results_dicts)

            _report({"type": "progress", "phase": "breakout", "status": "completed"})

        # ── Phase 6: AI Summary (disabled — guts kept for future use) ──
        ai_summary = None

        # ── Done ──────────────────────────────────────────────────────
        _report({"type": "progress", "phase": "complete", "status": "success"})

        return {
            "status":           "success",
            "region_detection": engine.region_result.to_dict() if engine.region_result else None,
            "summary":          _build_summary(results, engine, created_files),
            "ai_summary":       ai_summary,
            "ai_dirname":       ai_dirname,
            "output_path":      output_path,
            "results":          results_dicts,
        }

    except RegionDetectionError as e:

        return {
            "status": "error",
            "error": str(e),
            "suggestion": "Please provide region coordinates manually",
        }

    except Exception as e:
        import traceback
        log.error(f'Classification failed: {e}', exc=e)

        return {
            "status":    "error",
            "error":     f"Classification failed: {str(e)}",
            "traceback": traceback.format_exc(),
        }


def _run_breakout(
        results: List[Dict],
        pdf_path: str,
        output_path: str | None = None,
        on_progress: Callable | None = None,
        date_map: dict | None = None ) -> Dict:

    """Split source PDF into per-discipline files."""

    handler = BreakoutHandler(
        classification_results=results,
        pdf_path=pdf_path,
        output_dir=output_path,
    )

    return handler.breakout(on_progress=on_progress, date_map=date_map)


def _split_general_display(
        created_files: Dict,
        original_results: List[Dict] ) -> Dict:

    """Split merged Architectural entry back into Architectural + General display rows.

    BreakoutHandler produces one 'Architectural' PDF containing both.
    This restores separate rows for the UI DataGrid with individual
    page counts, both pointing to the same output file.
    """

    files = created_files.get('created_files', [])
    if not files:
        return created_files

    arch_entry = next((f for f in files if f['discipline'] == 'Architectural'), None)
    if not arch_entry:
        return created_files

    # Check if any General pages existed in the original classification
    general_indices = [
        r.get('page_index', r.get('page_num', 0))
        for r in original_results
        if (r.get('discipline') or r.get('category', 'Unknown')) == 'General'
    ]

    if not general_indices:
        return created_files

    arch_indices = [
        r.get('page_index', r.get('page_num', 0))
        for r in original_results
        if (r.get('discipline') or r.get('category', 'Unknown')) == 'Architectural'
    ]

    output_path = arch_entry['output_path']
    files.remove(arch_entry)

    # Architectural row (original A-sheets only)
    if arch_indices:
        files.append({
            'discipline':   'Architectural',
            'page_count':   len(arch_indices),
            'output_path':  output_path,
            'page_numbers': sorted(p + 1 for p in arch_indices),
        })

    # General row (original G-sheets, same output file)
    files.append({
        'discipline':   'General',
        'page_count':   len(general_indices),
        'output_path':  output_path,
        'page_numbers': sorted(p + 1 for p in general_indices),
    })

    return created_files


def _build_summary(
        results: list[PageResult],
        engine: ClassificationEngine,
        created_files: dict | None ) -> Dict:

    """Build summary statistics from classification results."""

    total_pages = len(results)

    # Counts by discipline
    discipline_counts: dict[str, int] = {}
    for r in results:
        disc = r.discipline or "Unknown"
        discipline_counts[disc] = discipline_counts.get(disc, 0) + 1

    # Counts by method
    method_counts: dict[str, int] = {}
    for r in results:
        method_counts[r.method] = method_counts.get(r.method, 0) + 1

    return {
        'total_pages':       total_pages,
        'discipline_counts': discipline_counts,
        'method_counts':     method_counts,
        'total_cost_usd':    round(engine.total_cost, 4),
        'timings':           engine.timings,
        'created_files':     created_files,
    }
