<#
.SYNOPSIS
    Builds SubCraft for Windows and packages it as a ready-to-run zip.

.DESCRIPTION
    Produces dist\SubCraft-<version>-windows-x64\ (SubCraft.exe, README, licenses) and
    dist\SubCraft-<version>-windows-x64.zip. The exe opens without a console window.
    Works in Windows PowerShell 5.1 and PowerShell 7.

.EXAMPLE
    .\scripts\build-windows.ps1
    .\scripts\build-windows.ps1 -SkipBuild    # package an existing dist build
#>
param(
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $match = Select-String -Path 'Cargo.toml' -Pattern '^version\s*=\s*"([^"]+)"' | Select-Object -First 1
    if (-not $match) { throw 'Could not read the version from Cargo.toml' }
    $version = $match.Matches[0].Groups[1].Value
    $onWindows = ($env:OS -eq 'Windows_NT')
    $exeName = if ($onWindows) { 'subcraft.exe' } else { 'subcraft' }

    if (-not $SkipBuild) {
        Write-Host "Building SubCraft $version (dist profile)..." -ForegroundColor Cyan
        cargo build --profile dist -p subcraft --features windows-gui
        if ($LASTEXITCODE -ne 0) { throw "cargo build failed (exit code $LASTEXITCODE)" }
    }

    $exe = Join-Path 'target\dist' $exeName
    if (-not (Test-Path $exe)) { throw "Build output not found: $exe" }

    $name = "SubCraft-$version-windows-x64"
    $dist = Join-Path $root 'dist'
    $stage = Join-Path $dist $name
    $zip = Join-Path $dist "$name.zip"
    $safeDist = [IO.Path]::GetFullPath($dist).TrimEnd('\') + '\'
    $safeStage = [IO.Path]::GetFullPath($stage)
    $safeZip = [IO.Path]::GetFullPath($zip)
    if (-not $safeStage.StartsWith($safeDist, [StringComparison]::OrdinalIgnoreCase) -or
        -not $safeZip.StartsWith($safeDist, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'Package paths must stay inside the workspace dist directory'
    }
    if (Test-Path -LiteralPath $safeZip) { Remove-Item -LiteralPath $safeZip -Force }
    New-Item -ItemType Directory -Path $stage -Force | Out-Null

    $outExe = if ($onWindows) { 'SubCraft.exe' } else { 'SubCraft' }
    Copy-Item $exe (Join-Path $stage $outExe)
    Set-Content -LiteralPath (Join-Path $stage 'PORTABLE') -Value 'Recovery copies are kept in this portable package.' -Encoding UTF8
    if (Test-Path -LiteralPath 'dist\Demo_Studio') {
        Copy-Item -LiteralPath 'dist\Demo_Studio' -Destination $stage -Recurse
    }
    foreach ($f in 'README.md', 'COMO_USAR_STUDIO.txt', 'LICENSE-MIT', 'LICENSE-APACHE') {
        Copy-Item $f $stage
    }
    $fonts = Join-Path $stage 'third-party'
    New-Item -ItemType Directory -Path $fonts -Force | Out-Null
    Copy-Item 'crates\ui\assets\fonts\*-LICENSE.txt' $fonts
    $studioDocs = Join-Path $stage 'docs'
    New-Item -ItemType Directory -Path (Join-Path $studioDocs 'screenshots') -Force | Out-Null
    Copy-Item 'docs\STUDIO.md' $studioDocs
    Copy-Item 'docs\screenshots\studio-main.png' (Join-Path $studioDocs 'screenshots')

    # Do not package Recovery or projects the user saved alongside the portable exe.
    $packageItems = @($outExe, 'README.md', 'COMO_USAR_STUDIO.txt', 'LICENSE-MIT', 'LICENSE-APACHE', 'PORTABLE', 'third-party', 'Demo_Studio', 'docs') |
        ForEach-Object { Join-Path $stage $_ } | Where-Object { Test-Path -LiteralPath $_ }
    Compress-Archive -LiteralPath $packageItems -DestinationPath $zip
    $size = '{0:N1} MB' -f ((Get-Item $zip).Length / 1MB)
    Write-Host ''
    Write-Host "Done: $zip ($size)" -ForegroundColor Green
    Write-Host "Run it: $(Join-Path $stage $outExe)"
}
finally {
    Pop-Location
}
