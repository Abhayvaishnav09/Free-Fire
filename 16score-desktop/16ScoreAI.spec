# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('src/score_ai/resources', 'score_ai/resources'), ('src/score_ai/detection/single-template', 'single-template'), ('16Score_logo.png', '.'), ('config.json', '.')],
    hiddenimports=['score_ai', 'score_ai.ui', 'score_ai.ui.main_window', 'score_ai.ui.pages', 'score_ai.ui.pages.camera_setup_pyqt', 'score_ai.ui.pages.home_page', 'score_ai.ui.pages.organization_page', 'score_ai.ui.pages.tournament_page', 'score_ai.ui.pages.match_page', 'score_ai.ui.pages.instruction_page', 'score_ai.ui.pages.streaming_source_page', 'score_ai.ui.components', 'score_ai.ui.components.header_widget', 'score_ai.ui.components.ui_components', 'score_ai.core', 'score_ai.core.api_service', 'score_ai.core.config', 'score_ai.core.config_manager', 'score_ai.utils', 'score_ai.utils.helpers', 'score_ai.utils.responsive_utils', 'score_ai.detection', 'score_ai.detection.killblocks', 'score_ai.detection.killfeed_detections', 'score_ai.resources', 'score_ai.resources.images', 'PyQt5', 'PyQt5.QtWidgets', 'PyQt5.QtGui', 'PyQt5.QtCore', 'PyQt5.sip', 'cv2', 'numpy', 'PIL', 'torch', 'torch._C', 'torch._utils', 'torch.autograd', 'torch.nn', 'torch.nn.functional', 'torch.optim', 'torch.utils', 'torch.utils.data', 'torch.utils.data.dataset', 'torch.utils.data.dataloader', 'torchvision', 'torchvision.transforms', 'torchvision.models', 'paddle', 'paddleocr', 'paddleocr.tools', 'paddleocr.tools.infer', 'paddleocr.tools.infer.predict_rec', 'paddleocr.tools.infer.predict_det', 'paddleocr.tools.infer.predict_cls', 'paddleocr.tools.infer.utility', 'paddleocr.ppocr', 'paddleocr.ppocr.utils', 'paddleocr.ppocr.utils.utility', 'paddleocr.ppocr.utils.visual', 'paddleocr.ppocr.utils.logging', 'paddleocr.ppocr.utils.save_load', 'paddleocr.ppocr.utils.character', 'paddleocr.ppocr.utils.cython_bbox', 'paddleocr.ppocr.utils.e2e_utils', 'paddleocr.ppocr.utils.get_image_file_list', 'paddleocr.ppocr.utils.network', 'paddleocr.ppocr.utils.postprocess', 'requests', 'aiohttp', 'asyncio', 'yarl', 'multidict', 'charset_normalizer', 'idna', 'typing_extensions', 'async_timeout', 'attrs', 'certifi', 'urllib3', 'scipy', 'scipy._lib', 'scipy.spatial', 'scipy.ndimage', 'scipy.signal', 'scipy.optimize', 'skimage', 'skimage.transform', 'skimage.measure', 'skimage.filters', 'skimage.feature', 'skimage.color', 'skimage.util', 'sklearn', 'sklearn.cluster', 'sklearn.metrics', 'rapidfuzz', 'rapidfuzz.fuzz', 'rapidfuzz.process', 'pkg_resources', 'pkg_resources.extern', 'setuptools', 'protobuf', 'grpc', 'grpcio', 'paddlex', 'paddlex.inference', 'paddlex.utils', 'paddlex.utils.config', 'paddlex.utils.device', 'paddlex.utils.deps', 'paddlex.utils.fonts', 'paddlex.utils.pipeline_arguments'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='16ScoreAI',
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
    icon=['16score_icon.ico'],
)
