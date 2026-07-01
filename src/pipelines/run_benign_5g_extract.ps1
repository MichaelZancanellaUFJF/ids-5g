<#
.SYNOPSIS
Pipeline de Extração e Calibração - Tráfego Benigno (Baseline 5G)

.DESCRIPTION
O tráfego benigno no laboratório 5G sofre distorções naturais (jitter, latência da RAN) 
que o diferenciam de uma rede cablada. Este script parametriza o 'pcap_to_csv.py' para 
extrair o tráfego do UE virtual, alinhando a mediana do Flow Duration 
($env:BENIGN_FD_MEDIAN_TARGET = "26.47") e forçando o recalculo de covariância.

Esta etapa garante que a classe majoritária (Benign / Classe 0) não dispare falsos 
positivos (False Alarm Rate) no IDS devido às características físicas do túnel GTP-U.
#>


# Benign 5G: extracao AGG + validacao LGBM (classe 0)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Pcap = if ($args.Count -ge 1) { $args[0] } else { "capture_benign_5g.pcap" }
$OutCsv = if ($args.Count -ge 2) { $args[1] } else { "capture_benign_5g_agg.csv" }

if (-not (Test-Path $Pcap)) {
    Write-Error "PCAP nao encontrado: $Pcap"
}

$env:AGG_N_ROWS = "20"
$env:AGG_BURST_IDLE_SEC = "0"
$env:NUMBER_MODE = "index_mean"
$env:FLOW_DURATION_MODE = "window"
$env:COVARIANCE_AGG_MODE = "max_run"
$env:VARIANCE_AGG_MODE = "cic_benign_min_std"
$env:GREIP_OGSTUN_FD_CALIB = "0"
$env:BENIGN_OGSTUN_CALIB = "1"
$env:BENIGN_URG_FROM_CIC = "1"
$env:BENIGN_URG_CALIB_SEED = "42"
$env:BENIGN_FD_MEDIAN_TARGET = "26.470675"
$env:BENIGN_COV_CALIB = "1"
$env:BENIGN_COV_FROM_CIC = "0"
$env:OGSTUN_PHYSICAL_CALIB = "1"
$env:OGSTUN_CALIB_LABEL = "BenignTraffic"
$Sched = if ($env:BENIGN_FD_SCHEDULE) { $env:BENIGN_FD_SCHEDULE } elseif (Test-Path "benign_schedule_seed42.json") { "benign_schedule_seed42.json" } else { "" }
if ($Sched) { $env:BENIGN_FD_SCHEDULE = $Sched }

Write-Host "[benign-5g] A extrair $Pcap -> $OutCsv"
python pcap_to_csv.py --in $Pcap --out-csv $OutCsv

Write-Host ""
Write-Host "[benign-5g] Validacao LGBM (classe 0):"
python validate_variant_lgbm.py benign $OutCsv
exit $LASTEXITCODE
