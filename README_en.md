# Framework for IoT Botnet Emulation and XAI-Based Calibration in 5G Networks

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Open5GS](https://img.shields.io/badge/5G_Core-Open5GS-brightgreen.svg)
![UERANSIM](https://img.shields.io/badge/RAN-UERANSIM-orange.svg)
![LightGBM](https://img.shields.io/badge/Machine_Learning-LightGBM-yellow.svg)
![SHAP](https://img.shields.io/badge/XAI-SHAP-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## About the Project

This repository contains a comprehensive framework for the **emulation, calibration, and injection of synthetic IoT botnet traffic in 5G networks**, developed within the context of Intrusion Detection Systems (IDS) research for **5G Core** environments.

The main objective of this project is to mitigate the **Data Shift** problem between public training datasets (such as **CICIoT2023**) and real-world experimentation topologies (such as the 5G Edge), utilizing **Explainable Artificial Intelligence (XAI)** and **Unsupervised Domain Adaptation** approaches.

Instead of simply replaying PCAP files, the framework generates synthetic traffic that is statistically consistent with the reference dataset, automatically calibrating its parameters based on the feature importance learned by a predictive Machine Learning model.

---

## Main Contributions

### 1. Stochastic Traffic Generation
The generator models the temporal and statistical behavior of botnets using Markov Chains (ON/OFF), AR(1) Auto-Regressive Models, Gaussian Mixture Models (GMM), and Zero-Inflated distributions for specific attack variants. This approach produces highly realistic physical flows rather than static PCAP replays.

### 2. XAI-Oriented Calibration
The framework uses a **LightGBM**-based IDS model to extract the spatial and temporal importance of features via **SHAP values (|φ|)**. The cost function minimizes the global **Wasserstein Distance (W₁)**, weighted by the importance of each variable. This ensures the alignment of critical physical signatures without causing heuristic overfitting.

### 3. Stochastic Bot Multiplexing (Volumetric Stress)
To validate the impact of massive volumetric attacks against Core network functions (UPF) without destroying the fidelity of temporal features (`flow_duration`, `IAT`), the generator implements bot multiplexing. Multiple synthetic streams are injected simultaneously, raising the Packets Per Second (PPS) rate to 5G Core saturation levels, while strictly preserving the individual flows' "physical envelope".

### 4. Real 5G Environment Injection
Once calibrated, the traffic is injected directly into UERANSIM's `uesimtun0` interface, traversing the entire Open5GS stack. It utilizes busy waiting and high-resolution timing to avoid artificial clustering caused by the OS scheduler.

### 5. Unsupervised Domain Adaptation (Zero-Shot Inference)
To ensure applicability at the 5G Edge without *Data Leakage*, the architecture implements an initial **Profiling Window**. The system uses strictly the first 10% of the unseen traffic (in an unsupervised manner) to extract the local statistical signature and recalibrate the scalar normalizer (*Test-Time Adaptation*). Following this profiling, the LightGBM model, with frozen weights from its original training, executes accurate **Zero-Shot** inference on the remaining 90% of the traffic.

---

## Project Architecture

```text
ids-5g/
│
├── data/
│   ├── pcaps/        # Real captures extracted from UPF (ogstun)
│   └── csvs/         # Features extracted via official pcap2csv
│
├── notebooks/
│   └── Visual validation (ECDF), SHAP, and Wasserstein Analysis
│
├── pipelines/
│   └── PowerShell scripts and ETL (PCAP → CSV)
│
├── requirements.txt
│
├── src/
│   ├── baseline/           # Statistical extraction of the source dataset
│   ├── calibration/        # Auto Calibration (Wasserstein + SHAP)
│   ├── domain_adaptation/  # Scaler calibration (Unsupervised) & Inference
│   ├── ids_model/          # Pre-trained LightGBM model
│   └── traffic_generator/  # Stochastic generator and LIVE 5G injector
│
└── testbed/
      Open5GS and UERANSIM configurations

      
Environment Reproduction
1. Initialize the 5G Testbed
Restart the Open5GS core and start the gNB and UE via UERANSIM to activate the uesimtun0 interface.

2. Automatic Calibration (Optional)
Bash
python3 src/calibration/auto_calibrate.py --variant greeth --real data/CIC_IoT_Dataset_Unificado_resumido.csv --phase global --iters 40
This updates the JSON profiles used by the generator.

3. Traffic Injection
Bash
sudo python3 src/traffic_generator/inject.py --variant greeth --profile live --iface uesimtun0
4. Capture and Feature Extraction
Bash
sudo tcpdump -i any -w capture.pcap
./pipelines/run_greeth_v6_extract.ps1 capture.pcap output.csv
5. Baseline Validation
Open the notebooks in notebooks/ to generate ECDFs, SHAP matrices, and statistical distances.

6. Domain Adaptation and Zero-Shot Inference