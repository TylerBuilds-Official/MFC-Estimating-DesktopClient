"""
Self-contained PDF classification logic with progress callbacks.

This module copies the core classify_pdf logic from tool_classify_pdf.py
but includes built-in progress callback support for desktop UI integration.

Imports only from mcp_server.classification (NOT mcp_server.tools) to avoid
triggering MCP server initialization.
"""
import os
import sys
import fitz
from typing import Dict, List, Optional, Literal, Callable

from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

# For PyInstaller: load .env from _internal folder
if getattr(sys, 'frozen', False):
    _exe_dir = os.path.dirname(sys.executable)
    load_dotenv(os.path.join(_exe_dir, '_internal', '.env'))
else:
    load_dotenv()

from . import debug_logger as log

from plan_classification import (
    RegionHandler,
    RegionDetectionError,
    BreakoutHandler,
    ClassificationEngine,
    OpenAIClassifier,
    ClaudeClassifier,
    PageClassification,
    AISummaryService,
    extract_image_from_region,
    optimize_image_for_api,
    get_pdf_page_count,
    bulk_extract_text_from_regions
)


# Parallel processing settings
DEFAULT_MAX_WORKERS = 25
DEFAULT_CALLS_PER_SECOND = 15.0  # 900/min


def classify_pdf(
    pdf_path: str,
    region: dict = None,
    auto_detect_region: bool = True,
    provider: Literal["openai", "anthropic", "auto"] = "auto",
    openai_api_key: str = None,
    anthropic_api_key: str = None,
    model: str = None,
    output_path: str = None,
    breakout_files: bool = True,
    calls_per_second: float = DEFAULT_CALLS_PER_SECOND,
    text_first: bool = True,
    batch_size: int = 15,
    max_concurrent: int = 8,
    progress_callback: Callable[[Dict], None] = None) -> Dict:
    """
    Classify pages in a construction drawing PDF by discipline.

    Uses text-first approach: extracts text from title block region and
    classifies via regex patterns. Falls back to batched AI vision only
    for pages where text extraction fails.

    This is the desktop-tooling version with built-in progress callbacks
    for UI integration.

    Args:
        pdf_path: Path to the PDF file
        region: Optional region dict with {x_ratio, y_ratio, w_ratio, h_ratio}
        auto_detect_region: If True and region=None, automatically detect region
        provider: "openai", "anthropic", or "auto" for AI fallback
        openai_api_key: OpenAI API key (uses OPENAI_APIKEY env var if not provided)
        anthropic_api_key: Anthropic API key (uses ANTHROPIC_API_KEY env var if not provided)
        model: Override default model
        output_path: Path to save breakout files
        breakout_files: If True, create separate PDFs per discipline
        calls_per_second: API rate limit (default 15.0)
        text_first: If True, try text extraction before AI (default True)
        batch_size: Number of images per AI API call (default 15)
        max_concurrent: Max batches to process in parallel (default 8)
        progress_callback: Callback(dict) for progress updates to UI

    Returns:
        dict with classification results and summary
    """

    # Load API keys from env if not provided
    if openai_api_key is None:
        openai_api_key = os.getenv('OPENAI_APIKEY')
    if anthropic_api_key is None:
        anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')

    if not os.path.exists(pdf_path):
        return {"error": f"PDF not found at {pdf_path}"}

    # Determine provider
    selected_provider = provider
    if provider == "auto":
        # Prefer OpenAI - it works better for title block recognition
        if openai_api_key:
            selected_provider = "openai"
        elif anthropic_api_key:
            selected_provider = "anthropic"
        else:
            return {"error": "No API key found. Set OPENAI_APIKEY or ANTHROPIC_API_KEY in .env"}

    # Validate we have the right key
    if selected_provider == "anthropic" and not anthropic_api_key:
        return {"error": "Anthropic provider selected but no ANTHROPIC_API_KEY found"}
    if selected_provider == "openai" and not openai_api_key:
        return {"error": "OpenAI provider selected but no OPENAI_APIKEY found"}

    # Helper to safely call progress callback
    def _report_progress(data: Dict):
        log.log_phase(data.get('phase', ''), data.get('status', ''), **{k: v for k, v in data.items() if k not in ('type', 'phase', 'status')})
        if progress_callback:
            progress_callback(data)

    try:
        #  PHASE 1: Region Detection
        region_info = None
        if region is None and auto_detect_region:
            _report_progress({"type": "progress", "phase": "region_detection", "status": "started"})

            # Region handler uses OpenAI (can update later)
            region_key = openai_api_key or anthropic_api_key
            region_handler = RegionHandler(openai_api_key=region_key)

            try:
                region_result = region_handler.auto_detect_region(pdf_path)
                region = region_result.region
                region_info = region_result.to_dict()
                print(f"[Region] Auto-detected using {region_result.method} (confidence: {region_result.confidence:.0%})")
                print(f"         x={region['x_ratio']:.2f}, y={region['y_ratio']:.2f}, w={region['w_ratio']:.2f}, h={region['h_ratio']:.2f}")
                if region_result.detected_samples:
                    print(f"         Samples: {region_result.detected_samples[:5]}")
                print(f"         Cache: ~/.pdfclassify_cache/region_cache.json")

                _report_progress({
                    "type": "progress",
                    "phase": "region_detection",
                    "status": "completed",
                    "method": region_result.method,
                    "confidence": region_result.confidence
                })

            except RegionDetectionError as e:
                return {
                    "error": str(e),
                    "suggestion": "Please provide region coordinates manually"
                }

        log.info(f'Provider: {selected_provider}')
        log.info(f'PDF: {pdf_path} | Output: {output_path}')
        log.info(f'Options: text_first={text_first}, breakout={breakout_files}, batch_size={batch_size}')

        if region is None:
            return {"error": "No region provided and auto_detect_region=False"}

        #  PHASE 2: Classification
        page_count = get_pdf_page_count(pdf_path)
        all_results: List[PageClassification] = []
        text_classified_count = 0
        needs_ai_pages: List[int] = []

        #  PHASE 2a: Text-First Classification
        if text_first:
            print(f"[Text] Extracting text from {page_count} pages...")
            text_map = bulk_extract_text_from_regions(pdf_path, region)

            engine = ClassificationEngine()
            processed_count = 0
            for page_num, text in text_map.items():
                processed_count += 1
                _report_progress({
                    "type": "progress",
                    "phase": "text_classification",
                    "current": processed_count,
                    "total": page_count
                })

                result, needs_ai = engine.classify_page_text_first(page_num, text)
                if result and not needs_ai:
                    # Convert ClassificationResult to PageClassification for consistency
                    all_results.append(PageClassification(
                        page_num=result.page_num,
                        category=result.category,
                        sheet_number=result.sheet_number,
                        confidence=result.confidence,
                        validated=True,
                        provider="text_regex",
                        cost_usd=0.0
                    ))
                    text_classified_count += 1
                    print(f"  [TEXT] Page {page_num + 1}: {result.sheet_number} -> {result.category}")
                else:
                    needs_ai_pages.append(page_num)

            print(f"[Text] Classified {text_classified_count}/{page_count} pages via text extraction")
            if needs_ai_pages:
                print(f"[AI]   {len(needs_ai_pages)} pages need AI fallback")
        else:
            needs_ai_pages = list(range(page_count))

        #  PHASE 2b: AI Classification for remaining pages
        ai_results = []
        provider_name = "text_only"

        if needs_ai_pages:
            # Create classifier based on provider
            if selected_provider == "anthropic":
                classifier = ClaudeClassifier(
                    api_key=anthropic_api_key,
                    model=model or "claude-sonnet-4-20250514",
                    calls_per_second=calls_per_second
                )
                provider_name = f"Anthropic ({classifier.model})"
            else:
                classifier = OpenAIClassifier(
                    api_key=openai_api_key,
                    model=model or "gpt-5-mini",
                    calls_per_second=calls_per_second
                )
                provider_name = f"OpenAI ({classifier.model})"

            print(f"[AI] Processing {len(needs_ai_pages)} pages via {provider_name} (batch_size={batch_size}, max_concurrent={max_concurrent})...")

            # Extract images ONLY for pages that need AI
            page_images = _extract_page_images_selective(pdf_path, region, needs_ai_pages)

            # Report AI classification starting
            ai_total = len(needs_ai_pages)
            _report_progress({
                "type": "progress",
                "phase": "ai_classification",
                "status": "started",
                "total": ai_total
            })

            # Progress callback fires as each batch completes
            def _ai_progress(completed, total):
                _report_progress({
                    "type": "progress",
                    "phase": "ai_classification",
                    "current": completed,
                    "total": total
                })

            # Use batched classification with parallel execution
            ai_results = classifier.classify_batch(
                page_images,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
                on_progress=_ai_progress
            )

            for result in ai_results:
                status = "OK" if result.validated else "??"
                print(f"  [AI] [{status}] Page {result.page_num + 1}: {result.sheet_number or 'N/A'} -> {result.category}")

            all_results.extend(ai_results)

        # Sort by page number
        all_results.sort(key=lambda r: r.page_num)
        results_dicts = [r.to_dict() for r in all_results]

        #  Create shared AI client (used for dirname + summary) 
        ai_client = None
        if selected_provider == "openai" and openai_api_key:
            ai_client = OpenAI(api_key=openai_api_key)
        elif selected_provider == "anthropic" and anthropic_api_key:
            ai_client = Anthropic(api_key=anthropic_api_key)

        #  AI Directory Naming 
        ai_dirname = None
        if ai_client and output_path:
            try:
                _report_progress({"type": "progress", "phase": "ai_dirname", "status": "started"})
                summary_service = AISummaryService(client=ai_client)
                pdf_filename = os.path.basename(pdf_path)
                dir_result = summary_service.create_dirname(pdf_filename)
                ai_dirname = dir_result.dir_name.strip()

                # Sanitize for filesystem safety
                ai_dirname = "".join(c for c in ai_dirname if c not in r'<>:"/\|?*').strip('. ')

                if ai_dirname:
                    output_path = os.path.join(output_path, ai_dirname)
                    os.makedirs(output_path, exist_ok=True)
                    print(f"[DirName] AI-generated output folder: {output_path}")

                _report_progress({"type": "progress", "phase": "ai_dirname", "status": "completed", "dirname": ai_dirname})

            except Exception as e:
                log.error(f'AI dirname failed: {e}', exc=e)
                print(f"[DirName] Failed (non-fatal): {e}")
                _report_progress({"type": "progress", "phase": "ai_dirname", "status": "failed", "error": str(e)})

        #  PHASE 3: Breakout Files
        created_files = {}
        if breakout_files:
            _report_progress({"type": "progress", "phase": "breakout", "status": "creating_files"})

            def _breakout_progress(current, total, category_name):
                _report_progress({
                    "type": "progress",
                    "phase": "breakout",
                    "status": "creating_files",
                    "current": current,
                    "total": total,
                    "category": category_name
                })

            created_files = _breakout_categories(
                results=results_dicts,
                pdf_path=pdf_path,
                output_path=output_path,
                on_progress=_breakout_progress
            )

            _report_progress({"type": "progress", "phase": "breakout", "status": "completed"})

        #  Build Summary
        summary = _build_summary(all_results, region_info, created_files, provider_name)

        #  Phase 4: AI Summary
        ai_summary = None
        try:
            if ai_client:
                _report_progress({"type": "progress", "phase": "ai_summary", "status": "started"})

                summary_service = AISummaryService(client=ai_client)
                validated = sum(1 for r in all_results if r.validated)
                confidence = validated / len(all_results) if all_results else 0.0
                result = summary_service.create_summary(json_data=summary, confidence_results=confidence)
                ai_summary = {
                    "text": result.summary,
                    "confidence": result.confidence
                }

                _report_progress({"type": "progress", "phase": "ai_summary", "status": "completed"})

        except Exception as e:
            log.error(f'AI summary failed: {e}', exc=e)
            print(f"[AI Summary] Failed (non-fatal): {e}")
            _report_progress({"type": "progress", "phase": "ai_summary", "status": "failed", "error": str(e)})

        _report_progress({"type": "progress", "phase": "complete", "status": "success"})

        return {
            "status": "success",
            "provider": provider_name,
            "region_detection": region_info,
            "region_used": region,
            "summary": summary,
            "ai_summary": ai_summary,
            "ai_dirname": ai_dirname,
            "output_path": output_path,
            "results": results_dicts
        }

    except Exception as e:
        import traceback
        log.error(f'Classification failed: {e}', exc=e)
        return {
            "error": f"Classification failed: {str(e)}",
            "traceback": traceback.format_exc()
        }


def _extract_page_images_selective(
    pdf_path: str,
    region: Dict,
    page_nums: List[int],
    zoom: float = 4.0
) -> List[tuple]:
    """
    Extract title block images ONLY for specified pages.

    More efficient than extracting all pages when only a subset needs AI.

    Args:
        pdf_path: Path to PDF file
        region: Region dict with ratios
        page_nums: List of page numbers to extract (0-indexed)
        zoom: Zoom factor for rendering

    Returns:
        List of (page_num, image_bytes) tuples
    """
    doc = fitz.open(pdf_path)
    page_images = []

    for page_num in page_nums:
        page = doc.load_page(page_num)
        image = extract_image_from_region(page, region, zoom=zoom)
        image_optimized = optimize_image_for_api(image, max_dimension=1500, quality=90)
        page_images.append((page_num, image_optimized))

    doc.close()
    return page_images


def _breakout_categories(results: List[Dict], pdf_path: str, output_path: str = None, on_progress: callable = None) -> Dict:
    """Create separate PDFs for each discipline"""
    handler = BreakoutHandler(
        classification_results=results,
        pdf_path=pdf_path,
        output_dir=output_path
    )
    return handler.breakout(on_progress=on_progress)


def _build_summary(
    results: List[PageClassification],
    region_info: Optional[Dict],
    created_files: Optional[Dict],
    provider_name: str) -> Dict:

    """Build summary statistics"""
    total_pages = len(results)

    # Count by category
    category_counts = {}
    for r in results:
        cat = r.category
        category_counts[cat] = category_counts.get(cat, 0) + 1

    # Classification method stats
    text_classified = sum(1 for r in results if r.provider == "text_regex")
    ai_classified = sum(1 for r in results if r.provider != "text_regex")

    # Validation stats
    validated_count = sum(1 for r in results if r.validated)
    failed_count = sum(1 for r in results if r.error)

    # Costs
    classification_cost = sum(r.cost_usd for r in results)
    region_cost = region_info.get('cost_usd', 0.0) if region_info else 0.0
    total_cost = classification_cost + region_cost

    return {
        'provider': provider_name,
        'total_pages': total_pages,
        'text_classified': text_classified,
        'ai_classified': ai_classified,
        'text_rate': f"{text_classified / total_pages * 100:.0f}%" if total_pages else "N/A",
        'validated_count': validated_count,
        'validation_rate': f"{validated_count / total_pages * 100:.0f}%" if total_pages else "N/A",
        'failed_count': failed_count,
        'category_counts': category_counts,
        'classification_cost_usd': round(classification_cost, 4),
        'region_detection_cost_usd': round(region_cost, 4),
        'total_cost_usd': round(total_cost, 4),
        'created_files': created_files
    }
