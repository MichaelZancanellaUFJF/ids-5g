<#
.SYNOPSIS
Pipeline de Extração e Validação AGG - Variante Mirai-GREETH

.DESCRIPTION
Este script automatiza a transformação do tráfego bruto (PCAP) da botnet Mirai-GREETH
em fluxos estatísticos agregados (CSV) prontos para a inferência do IDS LightGBM.

A variante GRE-ETH possui uma complexidade estrutural elevada devido ao overhead 
de encapsulamento. Para evitar o Data Shift, este pipeline injeta variáveis de ambiente 
rigorosas no extrator (pcap_to_csv.py), forçando o cálculo da Variância e Covariância 
('cic_greeth_min_std') a respeitar o envelope de agregação validado no baseline.

Após a extração, invoca o 'analyze_agg_profile.py' para auditar se o CSV resultante
atingiu os alvos de fidelidade estocástica exigidos pelo SHAP.
#>

# GREETH v6-G: extracao AGG + validacao de perfil (Windows / host)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Pcap = if ($args.Count -ge 1) { $args[0] } else { "capture_greeth_v6g.pcap" }
$OutCsv = if ($args.Count -ge 2) { $args[1] } else { "capture_greeth_agg_v6g.csv" }

if (-not (Test-Path $Pcap)) {
    Write-Error "PCAP nao encontrado: $Pcap"
}

$env:AGG_N_ROWS = "20"
$env:AGG_BURST_IDLE_SEC = "0"
$env:NUMBER_MODE = "index_mean"
$env:FLOW_DURATION_MODE = "window"
$env:COVARIANCE_AGG_MODE = "max_run"
$env:VARIANCE_AGG_MODE = "cic_greeth_min_std"
$env:GREIP_OGSTUN_FD_CALIB = "0"

Write-Host "[v6-G] A extrair $Pcap -> $OutCsv"
python pcap_to_csv.py --in $Pcap --out-csv $OutCsv

Write-Host ""
Write-Host "[v6-G] Validacao de perfil (alvos pre-inferencia):"
python analyze_agg_profile.py $OutCsv
$exitCode = $LASTEXITCODE

Write-Host ""
if ($exitCode -eq 0) {
    Write-Host "PASS - proximo passo:"
    Write-Host "  cd ..\Notebook"
    Write-Host "  # CAMINHO_CSV_TESTE = synthgen\$OutCsv"
    Write-Host "  python fog_validar_modelo.py"
}
else {
    Write-Host "FAIL - perfil nao atinge alvos v6-G."
    Write-Host "  Re-extrair o mesmo PCAP NAO altera Min/Cov (vem do inject)."
    Write-Host '  1) Copiar params_auto_greeth_v2.json v6-G + synth_generator.py para miraiserver'
    Write-Host '  2) run_greeth_v6_inject.sh -> NOVA captura ogstun (capture_greeth_v6g.pcap)'
    Write-Host '  3) .\run_greeth_v6_extract.ps1 .\capture_greeth_v6g.pcap capture_greeth_agg_v6g.csv'
    exit 1
}
