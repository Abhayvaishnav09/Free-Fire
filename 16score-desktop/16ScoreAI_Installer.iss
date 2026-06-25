; 16Score AI Desktop Application Installer
; Created with Inno Setup

#define MyAppName "16Score AI"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "16Score AI Team"
#define MyAppURL "https://16score.com"
#define MyAppExeName "16ScoreAI.exe"
#define MyAppIconName "16score_icon.ico"

[Setup]
; NOTE: The value of AppId uniquely identifies this application.
; Do not use the same AppId value in installers for other applications.
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile=
InfoBeforeFile=
InfoAfterFile=
OutputDir=installer_output
OutputBaseFilename=16ScoreAI_Setup
SetupIconFile=C:\Users\PRISHA\Desktop\16score-desktop\16score_icon.ico
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64
ArchitecturesInstallIn64BitMode=x64

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode

[Files]
; Main application executable
Source: "C:\Users\PRISHA\Desktop\16score-desktop\16ScoreAI\16ScoreAI.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\PRISHA\Desktop\16score-desktop\16ScoreAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; Application icon
Source: "C:\Users\PRISHA\Desktop\16score-desktop\16score_icon.ico"; DestDir: "{app}"; Flags: ignoreversion

; Additional resources and dependencies
Source: "C:\Users\PRISHA\Desktop\16score-desktop\config.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\PRISHA\Desktop\16score-desktop\requirements.txt"; DestDir: "{app}"; Flags: ignoreversion

; Player roster files — update these before each tournament and rebuild,
; OR replace them in {app} after installation without reinstalling.
; roster.json   : full player list with team groupings (used by fuzzy OCR matching)
; players.txt   : human-readable team sheet (same data, alternative format)
Source: "C:\Users\PRISHA\Desktop\16score-desktop\roster.json"; DestDir: "{app}"; Flags: ignoreversion
Source: "C:\Users\PRISHA\Desktop\16score-desktop\players.txt"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppIconName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[Code]
// Custom code to check system requirements
function InitializeSetup(): Boolean;
begin
  Result := True;
  
  // System checks removed for compatibility
end;

[Registry]
; Add application to Windows registry for proper uninstallation
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: string; ValueName: "DisplayName"; ValueData: "{#MyAppName}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: string; ValueName: "UninstallString"; ValueData: "{uninstallexe}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: string; ValueName: "DisplayIcon"; ValueData: "{app}\{#MyAppIconName}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: string; ValueName: "Publisher"; ValueData: "{#MyAppPublisher}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: string; ValueName: "DisplayVersion"; ValueData: "{#MyAppVersion}"; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: dword; ValueName: "NoModify"; ValueData: 1; Flags: uninsdeletekey
Root: HKLM; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\{#MyAppName}"; ValueType: dword; ValueName: "NoRepair"; ValueData: 1; Flags: uninsdeletekey

[CustomMessages]
; Custom messages for better user experience
english.LaunchAfterInstall=Launch 16Score AI after installation
english.InstallationComplete=Installation completed successfully!
english.WelcomeMessage=Welcome to 16Score AI Setup
english.InstallationDescription=This will install 16Score AI on your computer.
english.SystemRequirements=System Requirements:
english.Windows10Required=Windows 10 or later
english.DiskSpaceRequired=500MB free disk space
english.InternetRequired=Internet connection for updates

[Types]
Name: "full"; Description: "Full installation"
Name: "compact"; Description: "Compact installation"
Name: "custom"; Description: "Custom installation"; Flags: iscustom

[Components]
Name: "main"; Description: "16Score AI Application"; Types: full compact custom; Flags: fixed
Name: "resources"; Description: "Additional Resources"; Types: full custom
Name: "docs"; Description: "Documentation"; Types: full custom

[Dirs]
Name: "{app}\logs"; Permissions: users-full
Name: "{app}\config"; Permissions: users-full
Name: "{app}\temp"; Permissions: users-full

[UninstallDelete]
Type: files; Name: "{app}\logs\*"
Type: files; Name: "{app}\temp\*"
Type: dirifempty; Name: "{app}\logs"
Type: dirifempty; Name: "{app}\temp" 