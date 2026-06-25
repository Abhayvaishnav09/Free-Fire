"""
UI pages for the 16Score AI application.
"""
from score_ai.ui.pages.login_page import LoginPage
from score_ai.ui.pages.home_page import HomeScreen
from score_ai.ui.pages.organization_page import Screen2Page
from score_ai.ui.pages.tournament_page import TournamentPage
from score_ai.ui.pages.match_page import MatchPage
from score_ai.ui.pages.streaming_source_page import StreamingSourcePage
from score_ai.ui.pages.camera_setup_pyqt import CameraSetupPyQt
from score_ai.ui.pages.instruction_page import InstructionPage
from score_ai.ui.pages.settings_page import SettingsPage
from score_ai.ui.pages.forgot_password_page import ForgotPasswordPage
from score_ai.ui.pages.profile_page import ProfilePage
from score_ai.ui.pages.about_page import AboutPage

__all__ = [
    "LoginPage",
    "HomeScreen",
    "Screen2Page",
    "TournamentPage",
    "MatchPage",
    "StreamingSourcePage",
    "CameraSetupPyQt",
    "InstructionPage",
    "SettingsPage",
    "ForgotPasswordPage",
    "ProfilePage",
    "AboutPage",
] 