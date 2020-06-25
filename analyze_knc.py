"""
Analyze a KNC function and compute metrics.
"""

#!/usr/bin/python3

import sys
import math
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from logger import get_time
from importlib import import_module
from yaml import load, Loader

def get_result(run_name, superclass, result, outdir):
        """ Get the result value of a superclass """
        df = pd.read_csv(f"{outdir}/_results_{run_name}.csv", index_col=0)
        return df.loc[superclass, result]

def add_results(run_name, superclass, outdir ,**results):
    """ Append result columns in a superclass row """
    df = pd.read_csv(f"{outdir}/_results_{run_name}.csv", index_col=0)
    for resultname, result in results.items():
        df.loc[superclass, resultname] = result
    df.to_csv(f"{outdir}/_results_{run_name}.csv")

def read_knc_list(superclass, onemode, outdir):
    """ Read KNC plot values from csv file """
    df = pd.read_csv(f"{outdir}/{superclass}/{superclass}.{onemode}.knc.csv")
    return list(df.itertuples(index=False, name=None))

def compute_rc(knc_list, n_max, k_max):
    """ Compute representational consistency (AUC of KNC) based on connectivity measure """
    density_sum = 0
    if len(knc_list[0]) == 4: # Graph was build to compute connectivity measures
        ncomponents_sum = 0
        slcc_sum = 0
        for k in tqdm(range(0, k_max)):
            density_sum += knc_list[k][1]
            try:
                ncomponents_sum += (n_max - knc_list[k][2]) / (n_max - 1)
                slcc_sum += (knc_list[k][3] - 1) / (n_max - 1)
            except ZeroDivisionError:
                ncomponents_sum = 0
                slcc_sum = 0
        rc_density = (1 / k_max) * density_sum
        rc_ncomponents = (1 / k_max) * ncomponents_sum
        rc_slcc = (1 / k_max) * slcc_sum
        print(f"[Info] rc_density {rc_density:.8f}")
        print(f"[Info] rc_ncomponents {rc_ncomponents:.8f}")
        print(f"[Info] rc_slcc {rc_slcc:.8f}")
        return rc_density, rc_ncomponents, rc_slcc

    elif len(knc_list[0]) == 2: # Weight distribution was used to compute density
        for k in tqdm(range(0, k_max)):
            density_sum += knc_list[k][1]
        rc_density = (1 / k_max) * density_sum
        print(f"[Info] rc_density {rc_density:.8f}")
        return rc_density, None, None

def compute_avg_dens(knc_list, k_max):
    """ Compute the average density of a knc function with density connectivity measure """
    density_sum = 0
    for knc_step in knc_list:
        density_sum += knc_step[1]
    return density_sum / k_max

def get_k_0(knc_list):
    """ Return the k at which density becomes zero, Return 0 if density never becomes zero """
    for knc_step in knc_list:
        if knc_step[1] == 0:
            return knc_step[0]
    return math.inf

def get_rel_rc(run_name, superclass, rc_dens, onemode, outdir):
    """ Compute the representational consistency relative to its run superclass """
    df = pd.read_csv(f"{outdir}/_results_{run_name}.csv", index_col=0)
    if superclass == df.index[0]:
        return None
    else:
        return None
        if onemode == "t":
            rc_dens_super = df.iloc[0].rc_dens_t
        elif onemode == "b":
            rc_dens_super = df.iloc[0].rc_dens_b
        return rc_dens / rc_dens_super

def get_disc_nodes(run_name, classname, onemode, outdir):
    """ Get number of disconnected nodes in the onemode projection based on degree dist """
    nnodes = 0
    n_om = get_result(run_name, classname, f"n_{onemode}", outdir)
    with open(f"{outdir}/{classname}/{classname}.{onemode}.k.json", "r") as in_file:
        k_dist = json.load(in_file)
    for degree, count in k_dist.items():
        if int(degree) > 0:
            nnodes += count
    return n_om - nnodes

def get_median(run_name, classname, disttype, onemode, outdir):
    """ Get median of k (degree) or c (connectivity) distribution """
    labels = []
    with open(f"{outdir}/{classname}/{classname}.{onemode}.{disttype}.json", "r") as input_file:
        dist = json.load(input_file)
    for label, count in dist.items():
        labels.extend([int(label)] * count)
    return np.median(labels)

def get_weight_median(run_name, classname, onemode, outdir):
    """ Get median of w (edgeweight) distribution based on frequency table """
    m_om = get_result(run_name, classname, f"m_{onemode}", outdir)
    with open(f"{outdir}/{classname}/{classname}.{onemode}.w.json", "r") as input_file:
        w_dist = json.load(input_file)
    # Compute the 0.5 quantile
    quantile = 0.5
    quantilemargin = m_om * quantile
    counter = 0
    for weight, count in w_dist.items():
        if int(weight) > 0:
            counter += count
            if counter >= quantilemargin:
                return int(weight)
    return None

def get_stdev(run_name, classname, disttype, onemode, outdir):
    """ Get standard deviation of k (degree) or c (connectivity) distribution """
    labels = []
    with open(f"{outdir}/{classname}/{classname}.{onemode}.{disttype}.json", "r") as input_file:
        dist = json.load(input_file)
    for label, count in dist.items():
        labels.extend([int(label)] * count)
    return np.std(labels)

def analyze_knc(run_name, superclass, outdir):
    """ Get metrics of knc curve for both onemodes and save them in results file """
    n_t = int(get_result(run_name, superclass, "n_t", outdir))
    n_b = int(get_result(run_name, superclass, "n_b", outdir))
    print("here")
    knc_t = read_knc_list(superclass, "t", outdir)
    rc_t_dens, rc_t_ncomp, rc_t_slcc = compute_rc(knc_t, n_t, n_b)
    avg_dens_t = compute_avg_dens(knc_t, n_b)
    max_dens_t = knc_t[0][1]
    k_0_t = get_k_0(knc_t)
    rel_rc_t = get_rel_rc(run_name, superclass, rc_t_dens, "t", outdir)
    n_disc_t = get_disc_nodes(run_name, superclass, "t", outdir)
    k_med_t = get_median(run_name, superclass, "k", "t", outdir)
    c_med_t = get_median(run_name, superclass, "c", "t", outdir)
    w_med_t = get_weight_median(run_name, superclass, "t", outdir)
    k_sd_t = get_stdev(run_name, superclass, "k", "t", outdir)
    c_sd_t = get_stdev(run_name, superclass, "c", "t", outdir)
    # w_sd_t = ... # TODO: Compute mean/median/stdev based on frequency table

    knc_b = read_knc_list(superclass, "b", outdir)
    rc_b_dens, rc_b_ncomp, rc_b_slcc = compute_rc(knc_b, n_b, n_t)
    avg_dens_b = compute_avg_dens(knc_b, n_t)
    max_dens_b = knc_b[0][1]
    k_0_b = get_k_0(knc_b)
    rel_rc_b = get_rel_rc(run_name, superclass, rc_b_dens, "b", outdir)
    n_disc_b = get_disc_nodes(run_name, superclass, "b", outdir)
    k_med_b = get_median(run_name, superclass, "k", "b", outdir)
    c_med_b = get_median(run_name, superclass, "c", "b", outdir)
    w_med_b = get_weight_median(run_name, superclass, "b", outdir)
    k_sd_b = get_stdev(run_name, superclass, "k", "b", outdir)
    c_sd_b = get_stdev(run_name, superclass, "c", "b", outdir)
    # w_sd_b = ... # TODO: Compute mean/median/stdev based on frequency table

    add_results(run_name, superclass, outdir,
                rc_t_dens=rc_t_dens, rc_t_ncomp=rc_t_ncomp, rc_t_slcc=rc_t_slcc,
                avg_dens_t=avg_dens_t, max_dens_t=max_dens_t, k_0_t=k_0_t, rel_rc_t=rel_rc_t,
                n_disc_t=n_disc_t, k_med_t=k_med_t, c_med_t=c_med_t, w_med_t=w_med_t, k_sd_t=k_sd_t, c_sd_t=c_sd_t,
                rc_b_dens=rc_b_dens, rc_b_ncomp=rc_b_ncomp, rc_b_slcc=rc_b_slcc,
                avg_dens_b=avg_dens_b, max_dens_b=max_dens_b, k_0_b=k_0_b, rel_rc_b=rel_rc_b,
                n_disc_b=n_disc_b, k_med_b=k_med_b, c_med_b=c_med_b, w_med_b=w_med_b, k_sd_b=k_sd_b, c_sd_b=c_sd_b
                )

@get_time
def main():
    run_fn = sys.argv[1]
    run_name = run_fn.split("/")[-1].split(".")[0]
    run = load(open(run_fn).read(), Loader=Loader)
    outdir = run['config'].get('output_dir', 'out')
    
    for superclass in run['config']["classes"]:
        print("\n[Analyze knc]", superclass)
        try:
            analyze_knc(run_name, superclass, outdir)
        except FileNotFoundError as e:
            print(f"[Info] file not found {superclass} graph is the null graph\n{e}")
        except KeyError as e:
            print(f"[Info] key not found {superclass} graph is the null graph\n{e}")

    res = pd.read_csv(f"{outdir}/_results_{run_name}.csv", index_col=0)
    res = res.rename(columns={
        "k_t": "k_t_g", "k_b": "k_b_g",
        "k_t_om": "k_mean_t", "k_b_om": "k_mean_b",
        "c_t_om": "c_mean_t", "c_b_om": "c_mean_b",
        "rc_t_dens": "rc_dens_t", "rc_b_dens": "rc_dens_b",
        "rc_t_ncomp": "rc_ncomp_t", "rc_b_ncomp": "rc_ncomp_b",
        "rc_t_slcc": "rc_slcc_t", "rc_b_slcc": "rc_slcc_b",
        "avg_dens_t": "mean_dens_t", "avg_dens_b": "mean_dens_b",
        "max_dens_t": "dens_t", "max_dens_b": "dens_b",
        "n_disc_t": "ndisc_t", "n_disc_b": "ndisc_b",
        })
    cols = [ # Rearrange column order, Drop unwanted
        'n_t', 'n_b', 'm_g', 'dens_g', 'k_t_g', 'k_b_g', 'ndisc_t', 'ndisc_b',
        'm_t', 'dens_t', 'mean_dens_t', 'k_mean_t', 'k_med_t', 'k_sd_t', 'c_mean_t', 'c_med_t', 'c_sd_t','w_med_t',
        'k_0_t', 'rc_dens_t', 'rc_ncomp_t', 'rc_slcc_t', 'rel_rc_t',
        'm_b', 'dens_b', 'mean_dens_b', 'k_mean_b', 'k_med_b', 'k_sd_b', 'c_mean_b', 'c_med_b', 'c_sd_b','w_med_b',
        'k_0_b', 'rc_dens_b', 'rc_ncomp_b', 'rc_slcc_b', 'rel_rc_b',
        # 'superclass',
        ]
    res = res[cols]
    res.to_csv(f"{outdir}/_results_{run_name}.csv")


if __name__ == '__main__':
    main()
