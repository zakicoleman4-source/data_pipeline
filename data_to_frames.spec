# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for client_pipeline (Windows GUI).
# Build: pyinstaller data_to_frames.spec
# Output: dist/client_pipeline/  (run dist/client_pipeline/client_pipeline.exe)

from pathlib import Path
from PyInstaller.utils.hooks import (
    collect_submodules, collect_data_files, collect_dynamic_libs,
)

block_cipher = None

repo_root = Path(SPECPATH)

datas = [
    (str(repo_root / 'vendor' / 'ffmpeg' / 'bin'), 'vendor/ffmpeg/bin'),
    (str(repo_root / 'data_pipeline' / 'assets'), 'data_pipeline/assets'),
    # data_pipeline/configs/*.conf ships RTKLIB defaults the runtime
    # resolves via DEFAULT_CONF = configs / 'javad_avg_sp.conf'. Without
    # this entry the bundled exe raises FileNotFoundError on every run.
    (str(repo_root / 'data_pipeline' / 'configs'), 'data_pipeline/configs'),
    (str(repo_root / 'vendor' / 'android_rinex'), 'vendor/android_rinex'),
    # Bundle rnx2rtkp.exe so the client doesn't need RTKLIB-EX installed.
    # lab_tools._BUNDLE_FALLBACKS probes the bundle layout below before
    # PATH so the resolver returns the bundled binary by default.
    (str(repo_root / 'vendor' / 'rtklib'), 'vendor/rtklib'),
]

# scipy uses dynamic imports (scipy._lib._util, scipy.special._cdflib,
# scipy._lib.array_api_compat, scipy.linalg._fblas, ...) that PyInstaller
# misses if we list ``scipy`` as a single name. collect_submodules walks
# every public submodule + collect_data_files grabs the C extension data
# blobs. Same treatment for numpy.testing (numpy imports it at startup)
# and for our own data_pipeline package so the runtime hidden imports
# stay aligned with whatever we add to the source.
hiddenimports = (
    collect_submodules('scipy')
    + collect_submodules('numpy')
    + collect_submodules('data_pipeline')
)

# Some scipy / numpy submodules ship .pyd / .pyi data blobs alongside
# the python files — collect those too so the dynamic loaders find them.
datas += collect_data_files('scipy', include_py_files=False)
datas += collect_data_files('numpy', include_py_files=False)

# numpy + scipy C extensions depend on MKL / OpenBLAS / libomp DLLs that
# live next to the .pyd extension modules. collect_submodules captures
# the python side, collect_data_files grabs the .pyi/.dat blobs, but
# without collect_dynamic_libs the native dlls are missed and the exe
# crashes on `import numpy` with "DLL load failed while importing
# _multiarray_umath".
binaries = []
binaries += collect_dynamic_libs('numpy')
binaries += collect_dynamic_libs('scipy')

a = Analysis(
    ['data_to_frames_launcher.py'],
    pathex=[str(repo_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Test / dev tooling. ``unittest`` is NOT in this list — numpy.testing
        # imports it on initial module load and stripping it crashes the
        # whole import chain before main() ever runs.
        'pytest', 'tkinter.test',
        # Heavy unrelated packages that PyInstaller occasionally drags in
        # via transitive imports. None are used by data_pipeline; without
        # these excludes the dist balloons from ~250 MB to 5+ GB.
        'matplotlib', 'torch', 'torchvision', 'torchaudio', 'pyarrow',
        'onnxruntime', 'tensorflow', 'pandas', 'jupyter', 'notebook',
        'IPython', 'sympy', 'numba', 'sklearn', 'PIL', 'imageio',
        'rasterio', 'pyproj', 'shapely', 'fiona', 'geopandas',
        'h5py', 'tables', 'zarr', 'netCDF4',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe_gui = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='client_pipeline',
    icon=str(repo_root / 'app_icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,           # no console flash on double-click
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Console-mode twin so the same dist supports headless CLI use, scripts
# that pipe output, and smoke / verification runs without spawning the
# Tk window. Shares all binaries/data with the GUI exe -- the bootloader
# selects the right one from argv[0].
exe_cli = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='client_pipeline-cli',
    icon=str(repo_root / 'app_icon.ico'),
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe_gui,
    exe_cli,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='client_pipeline',
)
