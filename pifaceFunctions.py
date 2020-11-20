#!/usr/bin/env python
# coding: utf-8

import biotite
import biotite.structure as struc
from datetime import datetime
import requests
import h5py
import pandas as pd
import numpy as np
import os
from scipy import spatial
from scipy.spatial.distance import cdist
import random
import string


def sasa(dimer, chain1, chain2):
    """
    Biotite SASA calculation function for single chain and dimers.
    """
    
    total = 0

    atom_sasa = biotite.structure.sasa(dimer, vdw_radii="Single")
    res_sasa = biotite.structure.apply_residue_wise(dimer, atom_sasa, np.sum)
    dimer_sasa = np.nansum(res_sasa)

    chain_1 = dimer[dimer.chain_id == chain1]
    atom_sasa = biotite.structure.sasa(chain_1, vdw_radii="Single")
    res_sasa = biotite.structure.apply_residue_wise(chain_1, atom_sasa, np.sum)
    chain1_sasa = np.nansum(res_sasa)

    chain_2 = dimer[dimer.chain_id == chain2]
    atom_sasa = biotite.structure.sasa(chain_2, vdw_radii="Single")
    res_sasa = biotite.structure.apply_residue_wise(chain_2, atom_sasa, np.sum)
    chain2_sasa = np.nansum(res_sasa)

    total = chain1_sasa + chain2_sasa

    return chain1_sasa, chain2_sasa, dimer_sasa


def get_entries():
    """
    Read get entries.idx file from RCSB PDB FTP Server into a Pandas DataFrame
    """
    
    update_time = datetime.now().strftime("%Y%m%d%H%M%S")
    url = 'http://rsync.wwpdb.org/pub/pdb/derived_data/index/entries.idx'
    entries = requests.get(url)
    open('entries_%s.idx' % update_time, 'wb').write(entries.content)
    entriesidx_file = 'entries_%s.idx' % update_time
    
    entries_idx = []
    
    with open('entries_%s.idx' % update_time, 'r') as data:
        for line in data:
            line = line.replace('"', '')
            line.strip().replace('\n', '').split("\t")
            line = line.split("\t")
            line[-1] = line[-1].rstrip()
            entries_idx.append(line)

    entries_df = pd.DataFrame(entries_idx[2:])
    entries_df.columns = ["pdb_id", "header", "date", "compound", "source", "authors", "resolution", "exp_type"]
    entries_df = entries_df.set_index("pdb_id")
    entries_df["date"] = pd.to_datetime(entries_df.date).dt.date
    entries_df = entries_df.sort_values(by = ['date'])
    
    os.remove(entriesidx_file)
    
    return entries_df


def correctCIFLabels(cif_file):
    """
    As of Biotite Version 0.20.1 there is a problem with writing of cif files.
    There should be a white space at the end of _atom_site labels, which is missing.
    This function adds these missing white spaces.
    It might not be necessary in future versions of 
    """
    
    starting_file = open(cif_file, 'r')
    end_file = open(cif_file[:-4] + "_ok" + ".cif", 'a+')

    for line in starting_file:
        if line.startswith("_atom_site"):
            line = ''.join([line.strip(), " ", '\n'])
            end_file.write(line)
        else:
            end_file.write(line)
            continue
        
    starting_file.close()
    end_file.close()
    
    os.rename(cif_file[:-4] + "_ok" + ".cif", cif_file)


def extractInterface(dimer, c1, c2, vdwC, minC, nearC, mem):
    """
    This function extracts interfaces from provided dimers, by chain ids.
    """

    if os.path.exists("cache.hdf5"):
        os.remove("cache.hdf5")
    
    vdw_dict = {'ALA': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.165, 'C': 1.87, 'O': 1.55}, 'ARG': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 2.235, 'NE': 1.83, 'HE': 0.8, 'CZ': 1.87, 'NH1': 1.83, 'HH11': 0.6, 'HH12': 0.6, 'NH2': 1.83, 'HH21': 0.6, 'HH22': 0.6, 'C': 1.87, 'O': 1.55}, 'ARGN': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 2.235, 'NE': 1.83, 'HE': 0.8, 'CZ': 1.87, 'NH1': 1.83, 'HH11': 0.6, 'HH12': 0.6, 'NH2': 1.83, 'HH21': 0.6, 'C': 1.87, 'O': 1.55}, 'ASN': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 1.87, 'OD1': 1.55, 'ND2': 1.83, 'HD21': 0.8, 'HD22': 0.8, 'C': 1.87, 'O': 1.55, 'AD1': 1.55, 'AD2': 1.83}, 'ASP': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 1.87, 'OD1': 1.66, 'OD2': 1.66, 'C': 1.87, 'O': 1.55}, 'ASPH': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 1.87, 'OD1': 1.52, 'OD2': 1.55, 'HD': 0.8, 'C': 1.87, 'O': 1.55}, 'CYS': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'SG': 1.89, 'C': 1.87, 'O': 1.55}, 'GLN': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 1.87, 'OE1': 1.55, 'NE2': 1.83, 'HE21': 0.8, 'HE22': 0.8, 'C': 1.87, 'O': 1.55, 'AE1': 1.55, 'AE2': 1.83}, 'GLU': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 1.87, 'OE1': 1.66, 'OE2': 1.66, 'C': 1.87, 'O': 1.55}, 'GLUH': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 1.87, 'OE1': 1.52, 'OE2': 1.55, 'HE': 0.8, 'C': 1.87, 'O': 1.55}, 'GLY': {'N': 1.83, 'H': 0.8, 'CA': 2.235, 'C': 1.87, 'O': 1.55}, 'HIS': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'ND1': 1.72, 'HD1': 0.8, 'CD2': 2.1, 'NE2': 1.72, 'CE1': 2.1, 'C': 1.87, 'O': 1.55, 'AE1': 2.1, 'AE2': 1.72}, 'ILE': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.265, 'CG2': 2.165, 'CG1': 2.235, 'CD1': 2.165, 'C': 1.87, 'O': 1.55, 'CD': 2.165}, 'LEU': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.265, 'CD1': 2.165, 'CD2': 2.165, 'C': 1.87, 'O': 1.55}, 'LYS': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 2.235, 'CE': 2.235, 'NZ': 1.65, 'HZ': 0.6, 'HZ1': 0.6, 'HZ2': 0.6, 'HZ3': 0.6, 'C': 1.87, 'O': 1.55}, 'LYSN': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'CD': 2.235, 'CE': 2.235, 'NZ': 1.65, 'HZ1': 0.8, 'HZ2': 0.8, 'C': 1.87, 'O': 1.55}, 'MET': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'SD': 1.97, 'CE': 2.165, 'C': 1.87, 'O': 1.55}, 'PHE': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'CD1': 1.99, 'CD2': 1.99, 'CE1': 1.99, 'CE2': 1.99, 'CZ': 1.99, 'C': 1.87, 'O': 1.55}, 'PRO': {'N': 1.83, 'CD': 2.235, 'CA': 2.265, 'CB': 2.235, 'CG': 2.235, 'C': 1.87, 'O': 1.55}, 'SER': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'OG': 1.55, 'HG': 0.76, 'C': 1.87, 'O': 1.55}, 'THR': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.265, 'OG1': 1.55, 'HG1': 0.76, 'CG2': 2.165, 'C': 1.87, 'O': 1.55}, 'TRP': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'CD2': 2.04, 'CE2': 2.04, 'CE3': 1.99, 'CD1': 2.1, 'NE1': 1.72, 'HE1': 0.8, 'CZ2': 1.99, 'CZ3': 1.99, 'CH2': 1.99, 'C': 1.87, 'O': 1.55}, 'TYR': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'CD1': 1.99, 'CE1': 1.99, 'CD2': 1.99, 'CE2': 1.99, 'CZ': 2.04, 'OH': 1.55, 'HH': 0.76, 'C': 1.87, 'O': 1.55}, 'VAL': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.265, 'CG1': 2.165, 'CG2': 2.165, 'C': 1.87, 'O': 1.55}, 'HSC': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'CD2': 2.1, 'ND1': 1.72, 'HD1': 0.8, 'CE1': 2.1, 'NE2': 1.72, 'HE2': 0.8, 'C': 1.87, 'O': 1.55}, 'HSD': {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.235, 'CG': 2.04, 'ND1': 1.72, 'CE1': 2.1, 'CD2': 2.1, 'NE2': 1.72, 'HE2': 0.8, 'C': 1.87, 'O': 1.55}, 'ACE': {'C': 1.87, 'O': 1.55, 'CH3': 2.165}}
    default_dict = {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.265, 'C': 1.87, 'O': 1.55, 'CG': 2.265, 'CD': 2.235, 'NE': 1.83, 'HE': 0.8, 'C1': 1.87, 'C2': 1.87, 'C3': 1.87, 'CZ': 2.04, 'NH1': 1.83, 'H1': 0.8, 'H2': 0.8, 'H3': 0.8, 'H11': 0.8, 'H12': 0.8, 'H31': 0.8, 'H32': 0.8, 'HA' : 0.8, 'HA2' : 0.8, 'HA3' : 0.8, 'HB' : 0.8, 'HB1' : 0.8, 'HB2' : 0.8, 'HB3' : 0.8, 'HH2': 0.6, 'HH11': 0.6, 'HH12': 0.6, 'NH2': 1.83, 'HH21': 0.6, 'HH22': 0.6, 'OD1': 1.66, 'ND2': 1.83, 'HD21': 0.8, 'HD22': 0.8, 'AD1': 1.55, 'AD2': 1.83, 'OD2': 1.66, 'HD': 0.8, 'SG': 1.89, 'OE1': 1.66, 'NE2': 1.83, 'HE2': 0.8, 'HE3': 0.8, 'HE21': 0.8, 'HE22': 0.8, 'AE1': 2.1, 'AE2': 1.83, 'OE2': 1.66, 'ND1': 1.72, 'HD1': 0.8, 'HD2': 0.8, 'HD3': 0.8, 'HD11': 0.8, 'HD12': 0.8, 'HD13': 0.8, 'HD23': 0.8, 'CD2': 2.165, 'CE1': 2.1, 'CG2': 2.165, 'CG1': 2.235, 'CD1': 2.165, 'CE': 2.235, 'NZ': 1.65, 'HZ': 0.6, 'HZ1': 0.8, 'HZ2': 0.8, 'HZ3': 0.6, 'SD': 1.97, 'CE2': 2.04, 'OG': 1.55, 'HG': 0.76, 'OG1': 1.55, 'HG1': 0.76, 'HG2': 0.76, 'HG3': 0.76, 'HG11': 0.76, 'HG12': 0.76, 'HG13': 0.76, 'HG21': 0.76, 'HG22': 0.76, 'HG23': 0.76, 'CE3': 1.99, 'NE1': 1.72, 'HE1': 0.8, 'HO1': 0.8, 'HO2': 0.8, 'HO3': 0.8, 'CZ2': 1.99, 'CZ3': 1.99, 'CH2': 1.99, 'OH': 1.55, 'OXT': 1.55, 'O1': 1.55, 'O2': 1.55, 'O3': 1.55, 'HH': 0.76, 'HE2': 0.8, 'CH3': 2.165}
    no_digit_dict = {'N': 1.83, 'H': 0.8, 'CA': 2.265, 'CB': 2.265, 'C': 1.87, 'O': 1.55, 'CG': 2.265, 'CD': 2.235, 'NE': 1.83, 'HE': 0.8, 'CZ': 2.04, 'NH': 1.83, 'HA': 0.8, 'HB': 0.8, 'HH': 0.76, 'OD': 1.66, 'ND': 1.83, 'HD': 0.8, 'AD': 1.83, 'SG': 1.89, 'OE': 1.66, 'AE': 2.1, 'CE': 2.235, 'NZ': 1.65, 'HZ': 0.8, 'SD': 1.97, 'OG': 1.55, 'HG': 0.76, 'HO': 0.8, 'CH': 2.165, 'OH': 1.55, 'OXT': 1.55, 'D': 0.8, 'DZ': 0.8, 'HXT': 0.8, 'HAB': 0.8}

    chain_1 = dimer[dimer.chain_id == c1]
    chain_2 = dimer[dimer.chain_id == c2]

    chain_1_coord = chain_1.coord
    chain_2_coord = chain_2.coord

    try:
        chain_1_vdw = [float(vdw_dict[chain_1[i].res_name][chain_1[i].atom_name]) for i in range(len(chain_1))]

    except:
        chain_1_vdw = []

        for i in range(len(chain_1)):

            if chain_1[i].res_name in vdw_dict and chain_1[i].atom_name in vdw_dict[chain_1[i].res_name]:
                chain_1_vdw.append(float(vdw_dict[chain_1[i].res_name][chain_1[i].atom_name]))

            elif chain_1[i].atom_name in default_dict:
                chain_1_vdw.append(float(default_dict[chain_1[i].atom_name]))

            else:
                chain_1_vdw.append(float(no_digit_dict[chain_1[''.join([j for j in chain_1[i].atom_name if not j.isdigit()])].atom_name]))

    try:
        chain_2_vdw = [float(vdw_dict[chain_2[i].res_name][chain_2[i].atom_name]) for i in range(len(chain_2))]

    except:
        chain_2_vdw = []

        for i in range(len(chain_2)):

            if chain_2[i].res_name in vdw_dict and chain_2[i].atom_name in vdw_dict[chain_2[i].res_name]:
                chain_2_vdw.append(float(vdw_dict[chain_2[i].res_name][chain_2[i].atom_name]))

            elif chain_2[i].atom_name in default_dict:
                chain_2_vdw.append(float(default_dict[chain_2[i].atom_name]))

            else:
                chain_2_vdw.append(float(no_digit_dict[chain_2[''.join([j for j in chain_1[i].atom_name if not j.isdigit()])].atom_name]))

    if len(chain_1) * len(chain_2) * 8 > mem * (1024**3)/3:

        hdf5_file = True

        hdf5_store = h5py.File("cache.hdf5", "a")

        critdist_array = hdf5_store.create_dataset("critdist_array", (len(chain_1), len(chain_2)))
        n = len(chain_2)
        chunks = 10

        chunk_size = [[i * int(n / chunks), (i + 1) * int(n / chunks)] if i + 1 != chunks else [i * int(n / chunks), n]
                      for i in range(chunks)]

        for start, end in chunk_size:
            critdist_array[:, start:end] = np.array(chain_2_vdw[start:end]) + np.array(chain_1_vdw).reshape((-1, 1)) + vdwC

        realdist_array = hdf5_store.create_dataset('realdist_array', data=cdist(chain_1_coord, chain_2_coord, 'euclidean'))

        contacting_bool = np.zeros(realdist_array.shape, dtype=bool)

        n = len(realdist_array)
        chunks = 500

        chunk_size = [[i * int(n / chunks), (i + 1) * int(n / chunks)] if i + 1 != chunks else [i * int(n / chunks), n]
                      for i in range(chunks)]

        for start, end in chunk_size:
            contacting_bool[start:end, :] = realdist_array[start:end] <= critdist_array[start:end]

        contacting_1 = np.unique(chain_1[np.any(contacting_bool, axis=1)].res_id)
        contacting_2 = np.unique(chain_2[np.any(contacting_bool, axis=0)].res_id)

    else:

        hdf5_file = False

        realdist_array = cdist(chain_1_coord, chain_2_coord, 'euclidean')
        critdist_array = np.array(chain_2_vdw) + np.array(chain_1_vdw).reshape((-1, 1)) + vdwC

        contacting_1 = np.unique(chain_1[np.any(realdist_array <= critdist_array, axis=1)].res_id)
        contacting_2 = np.unique(chain_2[np.any(realdist_array <= critdist_array, axis=0)].res_id)

    if len(contacting_1) < minC or len(contacting_2) < minC:

        if hdf5_file is True:
            hdf5_store.close()
            os.remove("cache.hdf5")

        return None

    contacting = dimer[np.logical_or(np.logical_and(dimer.chain_id == c1, np.isin(dimer.res_id, contacting_1)), 
                                     np.logical_and(dimer.chain_id == c2, np.isin(dimer.res_id, contacting_2)))]

    contacting_ca = contacting[contacting.atom_name == 'CA']
    dimer_ca = dimer[dimer.atom_name == 'CA']
    pos_nearby_ca = dimer_ca[~struc.filter_intersection(dimer_ca, contacting_ca)]
    nearby_array = np.zeros((len(pos_nearby_ca), len(contacting_ca)))

    for i in range(len(pos_nearby_ca)):
        nearby_array[i] = struc.distance(pos_nearby_ca[i], contacting_ca)

    nearby_ca = pos_nearby_ca[np.any(nearby_array <= nearC, axis = 1)]

    nearby_1 = nearby_ca[nearby_ca.chain_id == c1].res_id
    nearby_2 = nearby_ca[nearby_ca.chain_id == c2].res_id

    nearby = dimer[np.logical_or(np.logical_and(dimer.chain_id == c1, np.isin(dimer.res_id, nearby_1)), 
                                 np.logical_and(dimer.chain_id == c2, np.isin(dimer.res_id, nearby_2)))]

    interface_1 = sorted(list(contacting_1) + list(nearby_1))
    interface_2 = sorted(list(contacting_2) + list(nearby_2))

    interface = dimer[np.logical_or(np.logical_and(dimer.chain_id == c1, np.isin(dimer.res_id, interface_1)), 
                                    np.logical_and(dimer.chain_id == c2, np.isin(dimer.res_id, interface_2)))]

    if hdf5_file is True:
        hdf5_store.close()
        os.remove("cache.hdf5")
    
    return ", ".join(map(str, contacting_1)), ", ".join(map(str, contacting_2)), ", ".join(map(str, nearby_1)), ", ".join(map(str, nearby_2)), interface


def readMMalignTrans(transfile):

    time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")

    with open(transfile, 'r') as trans_result:
        trans_list = trans_result.readlines()

    trans_matrix = ''.join(trans_list[1:5])

    return time_stamp, trans_matrix


def readMMalignResult(mmalignfile):

    with open(mmalignfile, 'r') as mmalign_result:
        result_list = mmalign_result.readlines()

    time_stamp = datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    interface_1 = result_list[7].split(":")[1].split(".")[0].strip()
    interface_2 = result_list[8].split(":")[1].split(".")[0].strip()
    aligned_length = int(result_list[12].split(",")[0].split("=")[1])
    rmsd = float(result_list[12].split(",")[1].split("=")[1])
    sequence_identity = float(result_list[12].split(",")[2].split("=")[2])
    tm_score_int1 = float(result_list[13].split("=")[1].split("(")[0])
    tm_score_int2 = float(result_list[14].split("=")[1].split("(")[0])
    aligned_residues = ''.join(result_list[18:21])

    return time_stamp, interface_1, interface_2, aligned_length, rmsd, sequence_identity, tm_score_int1, tm_score_int2, aligned_residues


def find_center(coords):
    return np.array([sum(coords[:, 0])/len(coords), sum(coords[:, 1])/len(coords), sum(coords[:, 2])/len(coords)])


def find_R(coords):
    candidates = coords[spatial.ConvexHull(coords).vertices]
    dist_mat = spatial.distance_matrix(candidates, candidates)
    i, j = np.unravel_index(dist_mat.argmax(), dist_mat.shape)
    return spatial.distance.euclidean(candidates[i], candidates[j])


def get_random_string(length):
    letters_and_digits = string.ascii_lowercase + string.digits
    return "".join([random.choice(letters_and_digits) for i in range(length)])
