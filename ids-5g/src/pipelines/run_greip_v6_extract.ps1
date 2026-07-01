<#
.SYNOPSIS
Pipeline de Extração Cirúrgica AGG - Variante Mirai-GREIP

.DESCRIPTION
Ao contrário do GRE-ETH, a variante GRE-IP exige um tratamento agressivo de fluxos curtos 
(~65% de fluxos com duração zero no dataset real). 
Este pipeline injeta a variável '$env:GREIP_FD_CALIB_MODE = "deterministic"' no 
extrator, instruindo o algoritmo de agregação a respeitar a anomalia de pacotes órfãos 
durante o cálculo das janelas temporais.

Termina com a invocação do validador focado na métrica 'Min', que é a variável 
mais sensível (maior peso SHAP) para a classificação desta ameaça.
#>

# GREIP v2: extracao AGG + validacao Min (Windows / host)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Pcap = if ($args.Count -ge 1) { $args[0] } else { "capture_greip_v2.pcap" }
$OutCsv = if ($args.Count -ge 2) { $args[1] } else { "capture_greip_agg_v2.csv" }

if (-not (Test-Path $Pcap)) {
    Write-Error "PCAP nao encontrado: $Pcap"
}

$env:AGG_N_ROWS = "20"
$env:AGG_BURST_IDLE_SEC = "0"
$env:NUMBER_MODE = "index_mean"
$env:FLOW_DURATION_MODE = "window"
$env:COVARIANCE_AGG_MODE = "max_run"
$env:VARIANCE_AGG_MODE = "cic_greip_min_std"
$env:GREIP_OGSTUN_FD_CALIB = "0"
$env:GREIP_FD_CALIB_MODE = "deterministic"

Write-Host "[greip-v2] A extrair $Pcap -> $OutCsv"
python pcap_to_csv.py --in $Pcap --out-csv $OutCsv

Write-Host ""
Write-Host "[greip-v2] Validacao de perfil (Min / classe 2):"
python analyze_greip_profile.py $OutCsv
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "PASS - proximo passo:"
    Write-Host "  cd ..\Notebook"
    Write-Host "  # CAMINHO_CSV_TESTE = synthgen\$OutCsv"
    Write-Host "  python fog_validar_modelo.py"
}
else {
    Write-Host "FAIL - perfil Min fora do alvo greip."
    Write-Host "  1) Copiar params_auto_greip_v2.json + synth_generator.py para miraiserver"
    Write-Host "  2) ./run_greip_v6_inject.sh -> NOVA captura ogstun"
    Write-Host "  3) .\run_greip_v6_extract.ps1 .\capture_greip_v2.pcap capture_greip_agg_v2.csv"
    exit 1
}
