# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for PlanClassification DesktopClient.

Build:
    pyinstaller pdf_classifier.spec

Output:
    dist/pdf_classifier/pdf_classifier.exe
"""

import os

block_cipher = None

a = Analysis(
    ['src/desktop_tooling/__main__.py'],
    pathex=[],
    binaries=[],
    datas=[
        # API keys injected by parent process at runtime — no .env bundled
    ],
    hiddenimports=[
        'desktop_tooling',
        'desktop_tooling.cli',
        'desktop_tooling.classifier',
        'plan_classification',
        'plan_classification.engine',
        'plan_classification.pipeline_config',
        'plan_classification.classify',
        'plan_classification.classify.sheet_classifier',
        'plan_classification.region',
        'plan_classification.region.region_handler',
        'plan_classification.breakout_handler',
        'plan_classification.utils',
        'plan_classification.utils.ai',
        'plan_classification.utils.pdf',
        'plan_classification.constants',
        'dotenv',
        'anthropic',
        'fitz',
        'fitz.fitz',
        'PIL',
        'PIL.Image',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='pdf_classifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console app - stdout/stderr used for IPC
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='pdf_classifier',
)
