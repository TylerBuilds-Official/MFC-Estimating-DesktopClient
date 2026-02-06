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
        # Bundle .env for API keys (placed in _internal at runtime)
        ('.env', '.'),
    ],
    hiddenimports=[
        'desktop_tooling',
        'desktop_tooling.cli',
        'desktop_tooling.classifier',
        'plan_classification',
        'plan_classification.engine',
        'plan_classification.ai_classifier',
        'plan_classification.pdf_utils',
        'plan_classification.region_handler',
        'plan_classification.breakout_handler',
        'dotenv',
        'openai',
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
