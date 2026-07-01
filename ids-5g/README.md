# 🛡️ Framework para Emulação de Botnets IoT e Calibração Baseada em XAI em Redes 5G

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Open5GS](https://img.shields.io/badge/5G_Core-Open5GS-brightgreen.svg)
![UERANSIM](https://img.shields.io/badge/RAN-UERANSIM-orange.svg)
![LightGBM](https://img.shields.io/badge/Machine_Learning-LightGBM-yellow.svg)
![SHAP](https://img.shields.io/badge/XAI-SHAP-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📖 Sobre o Projeto

Este repositório contém um framework completo para **emulação, calibração e injeção de tráfego sintético de botnets IoT em redes 5G**, desenvolvido no contexto de pesquisas sobre Sistemas de Detecção de Intrusão (IDS) para ambientes **5G Core**.

O principal objetivo do projeto é reduzir o problema de **Data Shift** entre bases públicas de treinamento (como o **CICIoT2023**) e ambientes reais de experimentação utilizando uma abordagem baseada em **Explainable Artificial Intelligence (XAI)**.

Ao invés de simplesmente reproduzir arquivos PCAP, o framework gera tráfego sintético estatisticamente consistente com o dataset de referência, calibrando automaticamente seus parâmetros segundo a importância das características aprendidas por um modelo de Machine Learning.

---

# 🔬 Principais Contribuições

## ✔ Geração Estocástica de Tráfego

O gerador modela o comportamento temporal e estatístico das botnets utilizando:

- Cadeias de Markov (ON/OFF)
- Modelos Auto-Regressivos AR(1)
- Misturas Gaussianas (Gaussian Mixture Models - GMM)
- Distribuições Zero-Inflated para variantes específicas

Essa abordagem produz fluxos muito mais realistas do que simples replays de PCAP.

---

## ✔ Calibração Orientada por Explainable AI (XAI)

O framework utiliza um modelo IDS baseado em **LightGBM** para obter as importâncias das features através dos valores **SHAP (|φ|)**.

Durante a calibração, apenas as variáveis realmente relevantes para o classificador são ajustadas.

A otimização minimiza iterativamente a:

**Distância de Wasserstein (W₁)**

entre os dados reais e os dados sintéticos.

---

## ✔ Injeção em Ambiente 5G Real

Após calibrado, o tráfego é injetado diretamente na interface:

```
uesimtun0
```

do UERANSIM, atravessando toda a pilha do Open5GS.

Para garantir precisão temporal na transmissão dos pacotes, o injetor utiliza:

- Busy Waiting
- Temporização em alta resolução
- Controle fino entre bursts de pacotes

evitando agrupamentos artificiais provocados pelo escalonador do sistema operacional.

---

# 🏗 Arquitetura do Projeto

```
ids-5g/
│
├── data/
│   ├── Dataset CICIoT2023
│   ├── Capturas PCAP
│   └── CSVs extraídos
│
├── notebooks/
│   ├── Análises exploratórias
│   ├── Validação visual
│   └── SHAP Analysis
│
├── pipelines/
│   ├── Scripts PowerShell
│   └── ETL PCAP → CSV
│
├── requirements.txt
│
├── src/
│   ├── baseline/
│   │     Extração estatística do CICIoT2023
│   │
│   ├── calibration/
│   │     Auto Calibration (Wasserstein + SHAP)
│   │
│   ├── ids_model/
│   │     Modelo LightGBM treinado
│   │
│   └── traffic_generator/
│         Gerador estocástico
│         Injetor de tráfego
│         Perfis JSON
│
└── testbed/
      Configurações Open5GS
      Configurações UERANSIM
```

---

#  Reprodução do Ambiente

## 1. Inicializar o Testbed 5G

Reinicie o núcleo Open5GS:

```bash
sudo systemctl restart \
open5gs-amfd \
open5gs-smfd \
open5gs-upfd
```

Inicie a gNB:

```bash
./nr-gnb -c testbed/UERANSIM/open5gs-gnb.yaml
```

Inicie o UE:

```bash
sudo ./nr-ue -c testbed/UERANSIM/open5gs-ue.yaml
```

Após a conexão, a interface

```
uesimtun0
```

estará disponível para injeção de tráfego.

---

## 2. Calibração Automática (Opcional)

Caso deseje reproduzir o processo de calibração:

```bash
python3 src/calibration/auto_calibrate.py \
    --variant greeth \
    --real data/CIC_IoT_Dataset_Unificado_resumido.csv \
    --phase global \
    --iters 40
```

Ao término da execução serão atualizados automaticamente os arquivos:

```
params_auto_*.json
```

utilizados pelo gerador.

---

## 3. Injeção do Tráfego

```bash
sudo python3 src/traffic_generator/inject.py \
    --variant greeth \
    --profile live \
    --iface uesimtun0
```

O tráfego será transmitido através do túnel GTP-U até a UPF do Open5GS.

---

## 4. Captura e Extração das Features

Realize a captura utilizando `tcpdump`:

```bash
sudo tcpdump -i any -w captura.pcap
```

Depois converta o PCAP para CSV utilizando um dos pipelines.

Exemplo:

```powershell
./pipelines/run_greeth_v6_extract.ps1 captura.pcap saida.csv
```

---

## 5. Validação

Abra o notebook:

```
notebooks/5G - IDS.ipynb
```

para gerar:

- ECDF
- Comparação estatística
- Distâncias Wasserstein
- Análise SHAP

permitindo verificar a fidelidade entre os dados reais e sintéticos.

---

#  Tecnologias Utilizadas

## Redes 5G

- Open5GS
- UERANSIM
- Scapy
- tcpdump

## Ciência de Dados

- Python
- Pandas
- NumPy
- SciPy
- Matplotlib

## Machine Learning

- LightGBM
- SHAP
- Optuna
- Scikit-Learn

---

# 📊 Fluxo Geral do Framework

```
               CICIoT2023
                     │
                     ▼
          Extração Estatística
                     │
                     ▼
        Gerador Estocástico Inicial
                     │
                     ▼
       Calibração (SHAP + Wasserstein)
                     │
                     ▼
      Parâmetros Ótimos (JSON)
                     │
                     ▼
        Geração do Tráfego Sintético
                     │
                     ▼
      Injeção na Interface uesimtun0
                     │
                     ▼
             Open5GS (UPF)
                     │
                     ▼
            Captura via tcpdump
                     │
                     ▼
            Extração das Features
                     │
                     ▼
          Comparação com CICIoT2023
```

---

# 📄 Licença

Este projeto foi desenvolvido para fins de pesquisa acadêmica em Sistemas de Detecção de Intrusão para Redes 5G e pode ser utilizado como base para estudos e experimentos científicos.

---

## 👨‍💻 Autor

**Michael Zancanella Barboza**

Programa de Pós-Graduação em Ciência da Computação  
Universidade Federal de Juiz de Fora (UFJF)

Área de pesquisa:

- Redes 5G
- Open5GS
- CUPS
- Network Slicing
- IDS
- Explainable AI
- Machine Learning
- Segurança em IoT
- NWDAF