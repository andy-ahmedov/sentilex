# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import os
import json
import hashlib
from datetime import datetime, timezone
import tempfile
import pymorphy2_dicts_ru
from PyInstaller.utils.hooks import copy_metadata

project_root = Path(SPECPATH).resolve().parent
block_cipher = None
onefile = os.environ.get("SENTILEX_ONEFILE", "0") == "1"
pymorph_data = Path(pymorphy2_dicts_ru.get_path())
metadata_dir = Path(tempfile.mkdtemp(prefix="sentilex-build-info-"))
build_info = metadata_dir / "build_info.json"
build_info.write_text(json.dumps({
    "build_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    "dictionary": "NRC EmoLex 2022", "edition": "nrc-ru-literary-v1",
}), encoding="utf-8")
source_metadata = json.loads((project_root / "scripts/data/NRC2022.SOURCE.json").read_text(encoding="utf-8"))
source_dictionary = project_root / "scripts/data/nrc2022/Russian-NRC-EmoLex.txt"
if not source_dictionary.is_file():
    raise SystemExit("Prepare private research data first: python tools/prepare_nrc2022.py")
if hashlib.sha256(source_dictionary.read_bytes()).hexdigest() != source_metadata["sha256"]:
    raise SystemExit("NRC 2022 resource checksum mismatch")
datas = [
    (str(build_info), "desktop"),
    (str(source_dictionary), "scripts/data/nrc2022"),
    (str(project_root / "scripts/data/nrc2022/README.txt"), "scripts/data/nrc2022"),
    (str(project_root / "scripts/data/NRC2022.SOURCE.json"), "scripts/data"),
    (str(project_root / "scripts/data/NRC2022.SOURCE.md"), "scripts/data"),
    (str(project_root / "scripts/data/NRC2022.sha256"), "scripts/data"),
    (str(project_root / "scripts/data/NRC2022.LITERARY.json"), "scripts/data"),
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
        name="SentiSoft-NRC2022-Literary",
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
        name="SentiSoft-NRC2022-Literary",
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
        name="SentiSoft-NRC2022-Literary",
    )
