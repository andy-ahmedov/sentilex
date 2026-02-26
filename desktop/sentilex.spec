# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import pymorphy2_dicts_ru
from PyInstaller.utils.hooks import copy_metadata

project_root = Path(SPECPATH).resolve().parent
block_cipher = None
onefile = os.environ.get("SENTILEX_ONEFILE", "0") == "1"
pymorph_data = Path(pymorphy2_dicts_ru.get_path())
datas = [
    (str(project_root / "desktop" / "ui" / "main.ui"), "desktop/ui"),
    (str(project_root / "desktop" / "style.qss"), "desktop"),
    (str(project_root / "scripts" / "RuSentilex-2017.txt"), "scripts"),
    (str(pymorph_data), "pymorphy2_dicts_ru/data"),
]
datas += copy_metadata("pymorphy2-dicts-ru")

analysis = Analysis(
    [str(project_root / "desktop" / "main.py")],
    pathex=[str(project_root), str(project_root / "scripts")],
    binaries=[],
    datas=datas,
    hiddenimports=["functions", "pkg_resources"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(analysis.pure, analysis.zipped_data, cipher=block_cipher)

if onefile:
    exe = EXE(
        pyz,
        analysis.scripts,
        analysis.binaries,
        analysis.zipfiles,
        analysis.datas,
        [],
        name="Sentilex",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        runtime_tmpdir=None,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        analysis.scripts,
        [],
        exclude_binaries=True,
        name="Sentilex",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        analysis.binaries,
        analysis.zipfiles,
        analysis.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name="Sentilex",
    )
