# -*- mode: python ; coding: utf-8 -*-
block_cipher = None

a = Analysis(['backend/run_server.py'], pathex=['.'], binaries=[], datas=[], hiddenimports=['backend.app.api.main'], hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], win_no_prefer_redirects=False, win_private_assemblies=False, cipher=block_cipher, noarchive=False)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz, a.scripts, a.binaries, a.zipfiles, a.datas, [], name='orchestrator', debug=False, bootloader_ignore_signals=False, strip=False, upx=True, console=True)
