from Feature_extraction import Feature_extraction
import time
import warnings
warnings.filterwarnings('ignore')

import os
from tqdm import tqdm
from multiprocessing import Process
import pandas as pd

if __name__ == '__main__':

    start = time.time()

    print("========== CIC IoT feature extraction ==========")

    pcapfiles = ['capture.pcap']  # Coloque aqui seus arquivos PCAP

    destination_directory = 'output/'
    os.makedirs(destination_directory, exist_ok=True)

    for pcap_file in pcapfiles:

        lstart = time.time()

        print(pcap_file)

        print(">>>> 1. Processing .pcap file.")

        fe = Feature_extraction()

        output_file = os.path.join(
            destination_directory,
            os.path.splitext(os.path.basename(pcap_file))[0]
        )

        p = Process(
            target=fe.pcap_evaluation,
            args=(pcap_file, output_file)
        )

        p.start()
        p.join()

        print(">>>> 2. Merging csv.")

        csv_subfiles = os.listdir(destination_directory)

        mode = 'w'

        for f in tqdm(csv_subfiles):

            try:

                d = pd.read_csv(os.path.join(destination_directory, f))

                d.to_csv(
                    pcap_file + '.csv',
                    header=(mode == 'w'),
                    index=False,
                    mode=mode
                )

                mode = 'a'

            except Exception as e:
                print(f"Erro ao ler {f}: {e}")

        print(">>>> 3. Cleaning temporary csv files.")

        for cf in csv_subfiles:

            os.remove(os.path.join(destination_directory, cf))

        print(
            f'done! ({pcap_file}) '
            f'({round(time.time()-lstart,2)} s)'
        )

    end = time.time()

    print(f'Elapsed Time = {round(end-start,2)} s')