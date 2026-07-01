import numpy as np
import itertools
from scipy import stats

class Dynamic_features:
    def dynamic_calculation(self,ethsize):
        sum_packets = sum(ethsize)
        min_packets = min(ethsize)
        max_packets = max(ethsize)
        mean_packets = sum_packets / len(ethsize)
        std_packets = np.std(ethsize)

        return sum_packets,min_packets,max_packets,mean_packets,std_packets

    def dynamic_count(self,protcols_count):   #calculates the Number feature
        packets = 0
        for k in protcols_count.keys():
            packets = packets + protcols_count[k]

        return packets

    def dynamic_two_streams(self,incoming, outgoing):

        inco_ave = sum(incoming) / len(incoming)
        outgoing_ave = sum(outgoing) / len(outgoing)
        magnite = (inco_ave + outgoing_ave) ** 0.5

        inco_var = np.var(incoming)
        outgo_var = np.var(outgoing)
        radius = (inco_var + outgo_var) ** 0.5
        n_pairs = min(len(incoming), len(outgoing))
        if n_pairs >= 2 and inco_var > 0 and outgo_var > 0:
            correlation, _ = stats.pearsonr(incoming[:n_pairs], outgoing[:n_pairs])
        else:
            correlation = 0.0

        if n_pairs > 0:
            covariance = sum(
                (a - inco_ave) * (b - outgoing_ave)
                for (a, b) in zip(incoming[:n_pairs], outgoing[:n_pairs])
            ) / n_pairs
        else:
            covariance = 0.0
        var_ratio = 0
        if outgo_var != 0:
            var_ratio = inco_var / outgo_var

        weight = len(incoming) * len(outgoing)

        return magnite, radius, correlation, covariance, var_ratio, weight

