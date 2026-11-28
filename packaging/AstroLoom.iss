; packaging/AstroLoom.iss - Instalador Inno Setup 6 para AstroLoom (one-dir, por usuario)
; Compilar tras `pyinstaller --clean Astroloom.spec`:
;   iscc /DMyAppVersion=0.19.3 packaging\AstroLoom.iss

#ifndef MyAppVersion
  #define MyAppVersion "0.19.3"
#endif

#define MyAppName "AstroLoom"
#define MyAppPublisher "AstroLoom"
#define MyAppExeName "AstroLoom.exe"

[Setup]
AppId={{8D4F6A2E-5C31-4B9E-A872-3F2E91C4B710}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
; Instalación por usuario (%LOCALAPPDATA%\Programs\AstroLoom): sin UAC y carpeta escribible para models/, logs/ y config.json
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
OutputDir=..\dist
OutputBaseFilename=AstroLoom-Setup-x64
Compression=lzma2/normal
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\AstroLoom\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
