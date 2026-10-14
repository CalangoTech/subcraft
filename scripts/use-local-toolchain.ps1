# Dot-source this file when using the portable development toolchain on Windows:
# . .\scripts\use-local-toolchain.ps1
$taskRoot = Split-Path $PSScriptRoot -Parent
$taskTools = Join-Path $taskRoot '.tools'
if (Test-Path -LiteralPath (Join-Path $taskTools 'cargo\bin\cargo.exe')) {
    $env:CARGO_HOME = Join-Path $taskTools 'cargo'
    $env:RUSTUP_HOME = Join-Path $taskTools 'rustup'
    $env:TEMP = Join-Path $taskTools 'tmp'
    $env:TMP = $env:TEMP
    $env:PATH = "$(Join-Path $taskTools 'cargo\bin');$(Join-Path $taskTools 'mingw64\bin');$env:PATH"
    New-Item -ItemType Directory -Path $env:TEMP -Force | Out-Null
}
