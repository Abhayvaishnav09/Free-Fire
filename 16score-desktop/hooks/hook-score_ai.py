
from PyInstaller.utils.hooks import collect_all

# Collect all modules and data files for score_ai
datas, binaries, hiddenimports = collect_all('score_ai')

# Add specific modules that might be missed
hiddenimports += [
    'score_ai.ui',
    'score_ai.ui.main_window',
    'score_ai.ui.pages',
    'score_ai.ui.pages.login_page',
    'score_ai.ui.pages.home_page',
    'score_ai.ui.pages.organization_page',
    'score_ai.ui.pages.tournament_page',
    'score_ai.ui.pages.match_page',
    'score_ai.ui.pages.streaming_source_page',
    'score_ai.ui.pages.camera_setup_pyqt',
    'score_ai.ui.components',
    'score_ai.ui.components.header_widget',
    'score_ai.ui.components.ui_components',
    'score_ai.core',
    'score_ai.core.api_service',
    'score_ai.core.config',
    'score_ai.core.config_manager',
    'score_ai.detection',
    'score_ai.detection.killblocks',
    'score_ai.detection.killfeed_detections',
    'score_ai.utils',
    'score_ai.utils.helpers',
    'score_ai.utils.responsive_utils',
]
