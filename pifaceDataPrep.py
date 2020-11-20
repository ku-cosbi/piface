#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import requests
import traceback
import os
import sys
import glob
import argparse
import sqlite3
import pifaceFunctions
from itertools import combinations
from scipy import spatial
from datetime import datetime
import biotite.structure as struc
import biotite.structure.io as strucio
import biotite.database.rcsb as rcsb
from biotite.structure.io.pdbx import PDBxFile, set_structure


# Take user arguments

arg_parser = argparse.ArgumentParser(description='This is the Data Preparation component for PIFace Workflow. It is used for building/updating; -Fasta, CIF tables in database using RCSB PDB -Finding dimers from CIF files -Extracting interfaces from dimer files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
new_res_group = arg_parser.add_mutually_exclusive_group()
new_res_group.add_argument("-r", "--restart", default=False, action="store_true", help="whether the run is a restart or not. if it is an automatic restart (e.g. cluster failure, auto resubmission) --restart is set by default by checking the availability of current_config.log")
new_res_group.add_argument("-n", "--new", default=False, action="store_true", help="whether the run is a new PIFace build or an update")
new_res_group.add_argument("-e", "--error", default=False, action="store_true", help="try to run for PDB IDs that has a processed code = 2.")
new_res_group.add_argument("-c", "--current", default=False, action="store_true", help="run for current PDB IDs that has a processed code = 0, without changing entries table")
arg_parser.add_argument("-f", "--file", default=False, help="provide a file with PDB IDs instead of downloading the latest entries.idx")
arg_parser.add_argument("-p", "--prefix", default=os.getcwd(), help="run the program in a different directory than the default (give exact path of the custom directory)")
arg_parser.add_argument("-y", "--yes", default=False, action="store_true", help="answer \"Are you in an existing PIface or a clean directory?\" from command line")
arg_parser.add_argument("-db", "--database", default="piface", help="give a different name than the default to the database. .db is automatically added.")

arg_parser.add_argument("-S", "--sasa", default=1.0, help="assign custom SASA difference between single chain SASA summation and dimer SASA for dimer identification. this option will be IGNORED if the run is an update of an already existing database.")
arg_parser.add_argument("-V", "--vdw", default=0.5, help="assign distance for identification of contacting residues. this option will be IGNORED if the run is an update of an already existing database.")
arg_parser.add_argument("-M", "--mincontacting", default=5, help="assign minimum number of residues per chain for interfaces. this option will be IGNORED if the run is an update of an already existing database.")
arg_parser.add_argument("-N", "--nearby", default=6.0, help="assign distance value for identification of nearby residues. this option will be IGNORED if the run is an update of an already existing database.")

arg_parser.add_argument("-m", "--memory", default=16.0, help="maximum amount of memory (RAM) usage by the program")

args = arg_parser.parse_args()


# Check if all the needed directories, databases and files are available. If not raise helpful error messages to identify the problem. Also gather other necessary files.

workDir = args.prefix
logsDir = "%s/logs" % workDir
args.database = "%s.db" % args.database

update_time = datetime.now().strftime("%Y%m%d%H%M%S")

args.sasa = float(args.sasa)
args.vdw = float(args.vdw)
args.mincontacting = int(args.mincontacting)
args.nearby = float(args.nearby)
args.memory = float(args.memory)

# Make sure the user checks that they are in the correct directory. If --yes argument is given while running the code this part is skipped.

while not args.yes:
    inp = input("Code will run here: %s\nIs this an existing PIface or a clean directory? [yes (y) / no (n)]\n" % workDir)
    if (inp == "yes") or (inp == "y"):
        args.yes = True
        continue
    elif (inp == "no") or (inp == "n"):
        raise SystemExit("Please start the code in a PIface directory or a clean directory.\nIf you are running the code on a cluster, you can answer this question from command line with --yes option.")
    else:
        inp = input("Please provide a valid input.\nCode will run here: %s\nIs this an existing PIface or a clean directory? [yes (y) / no (n)]\n? [yes (y) / no (n)]" % workDir)

# Check if it is an update.

if args.new is False:
    if not os.path.exists("%s/%s" % (workDir, args.database)):
        raise SystemExit("PIface database not found:\n%s/%s" % (workDir, args.database))
    if (args.file is not False) and (os.path.exists(args.file) is False):
        raise SystemExit("Custom PDB List file not found:\n%s" % args.file)
    if not os.path.isdir(logsDir):
        raise SystemExit("Logs directory for existing DB cannot be found:\n%s" % logsDir)

    # Check if database correctly set up

    pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
    pi_cur = pi_conn.cursor()

    pi_cur.execute('SELECT name from sqlite_master WHERE type=\'table\';')
    pi_tables = [i[0] for i in pi_cur.fetchall()]
    req_tables = ['entries', 'fasta', 'cif', 'dimers', 'dimers_cif', 'interfaces', 'interfaces_cif', 'interfaces_seq']

    if set(pi_tables) != set(req_tables):
        pi_conn.close()
        raise SystemExit("Provided PIface DB tables are not correctly set up.\nExpected: %s\nFound: %s" % (req_tables, pi_tables))

    pi_cur.execute('SELECT name from sqlite_master WHERE type=\'index\';')
    pi_index = [i[0] for i in pi_cur.fetchall()]
    req_index = ['idx_entries', 'idx_entries_updatetime', 'idx_fasta', 'idx_cif', 'idx_dimers', 'idx_dimers_cif', 'idx_interfaces', 'idx_interfaces_cif', 'idx_interfaces_seq']

    if set(pi_index) != set(req_index):
        pi_conn.close()
        raise SystemExit("Provided PIface DB indexes are not correctly set up.\nExpected: %s\nFound: %s" % (req_index, pi_index))

    pi_conn.close()

    # Check settings of the existing database from config file (SASA Criteria, VDW Criteria, Min Contacting Criteria, Nearby Criteria)

    # Check if current_config.log exists and read it for an automatic restart if True.

    if os.path.exists("%s/current_config.log" % logsDir):
        with open("%s/current_config.log" % logsDir) as config:
            config_list = [i.split(":")[1].strip() for i in config.readlines()[1:]]

        args.restart = True
        update_time = config_list[0]
        args.database = config_list[2]
        args.new = False
        args.yes = True
        if config_list[4] == "False":
            args.file = False
        else:
            args.file = config_list[4]
        args.error = config_list[5]
        args.sasa = float(config_list[6])
        args.vdw = float(config_list[7])
        args.mincontacting = int(config_list[8])
        args.nearby = float(config_list[9])
        args.memory = float(config_list[10])

        print("\n--- %s ---\nRestarting from existing config.log file:\n%s/current_config.log\n" % (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logsDir))

    elif len(glob.glob("%s/config_*" % logsDir)) == 0:
        raise SystemExit("Previous configuration file cannot be found:\n%s/config_**.log" % logsDir)

    else:
        last_config = sorted(glob.glob("%s/config_*" % logsDir))[-1]
        with open(last_config) as config:
            config_list = [i.split(":")[1].strip() for i in config.readlines()[1:]]

        args.sasa = float(config_list[6])
        args.vdw = float(config_list[7])
        args.mincontacting = int(config_list[8])
        args.nearby = float(config_list[9])
        args.memory = float(config_list[10])

# Check if it is a new PIface build.

if args.new is True:
    if os.path.exists("%s/%s" % (workDir, args.database)):
        raise SystemExit("There is already a database:\n%s/%s\nThis might not be a clean directory." % (workDir, args.database))

    if os.path.isdir(logsDir):
        raise SystemExit("There is already a logs directory:\n%s\nThis might not be a clean directory." % logsDir)

    # Create logs directory and database from scratch with all the tables

    os.mkdir(logsDir)

    pi_conn = sqlite3.connect('%s/%s' % (workDir, args.database))
    pi_cur = pi_conn.cursor()
    pi_cur.execute(""" PRAGMA journal_mode=WAL; """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS entries (
                        pdb_id text NOT NULL,
                        header text NOT NULL,
                        date text NOT NULL,
                        compound text NOT NULL,
                        source text NOT NULL,
                        authors text NOT NULL,
                        resolution real NOT NULL,
                        exp_type text NOT NULL,
                        processed integer NOT NULL,
                        update_time text NOT NULL
                   ); """)
    pi_cur.execute(""" CREATE INDEX idx_entries ON entries(pdb_id); """)
    pi_cur.execute(""" CREATE INDEX idx_entries_updatetime ON entries(update_time); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS fasta (
                        pdb_id text NOT NULL,
                        entity_id text NOT NULL,
                        chain_id text NOT NULL,
                        info text,
                        source_organism text,
                        ncbi_taxonomy text,
                        sequence text,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE INDEX idx_fasta ON fasta(pdb_id, entity_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS cif (
                        pdb_id text NOT NULL,
                        chain_count integer,
                        combination_count integer,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_cif ON cif(pdb_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS dimers (
                        dimer_id text NOT NULL,
                        pdb_id text NOT NULL,
                        chain_1 text NOT NULL,
                        chain_2 text NOT NULL,
                        chain_1_sasa numeric NOT NULL,
                        chain_2_sasa numeric NOT NULL,
                        dimer_sasa numeric NOT NULL,
                        sasa_diff numeric NOT NULL,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_dimers ON dimers(dimer_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS dimers_cif (
                        dimer_id text NOT NULL,
                        cif_file text NOT NULL,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_dimers_cif ON dimers_cif(dimer_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS interfaces (
                        dimer_id text NOT NULL,
                        pdb_id text NOT NULL,
                        chain_1 text NOT NULL,
                        chain_2 text NOT NULL,
                        chain_1_contacting text NOT NULL,
                        chain_2_contacting text NOT NULL,
                        chain_1_nearby text NOT NULL,
                        chain_2_nearby text NOT NULL,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_interfaces ON interfaces(dimer_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS interfaces_cif (
                        dimer_id text NOT NULL,
                        cif_file text NOT NULL,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_interfaces_cif ON interfaces_cif(dimer_id); """)

    pi_cur.execute(""" CREATE TABLE IF NOT EXISTS interfaces_seq (
                        dimer_id text NOT NULL,
                        pdb_id text NOT NULL,
                        chain_1 text NOT NULL,
                        chain_2 text NOT NULL,
                        sequence_1 text NOT NULL,
                        sequence_2 text NOT NULL,
                        insert_time text NOT NULL
                    ); """)
    pi_cur.execute(""" CREATE UNIQUE INDEX idx_interfaces_seq ON interfaces_seq(dimer_id); """)

    pi_conn.commit()
    pi_conn.close()

# Write run configurations to a current_config file. current_config will be moved to config_"update_date".log when the run is completely finished.  

if (args.restart is False) or ((os.path.exists("%s/current_config.log" % logsDir) is False) and (args.restart is True)):
    with open("%s/current_config.log" % logsDir, "a+") as f:
        f.write("--- %s ---\n" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        f.write("Update Time: %s\n" % update_time)
        f.write("Prefix: %s\n" % workDir)
        f.write("Database Name: %s\n" % args.database)
        f.write("New PIface DB: %s\n" % args.new)
        f.write("Custom PDB List File: %s\n" % args.file)
        f.write("Error Check: %s\n" % args.error)
        f.write("SASA Value: %s\n" % args.sasa)
        f.write("VDW Criteria for Contacting Residues: %s\n" % args.vdw)
        f.write("Minimum Number of Residues on Interface Chains: %s\n" % args.mincontacting)
        f.write("Distance Criteria for Nearby Residues: %s\n" % args.nearby)
        f.write("Maximum Memory Usage: %s\n" % args.memory)

    print("\n--- %s ---\nRun is ready to start.\nAll starting configurations are written to config file:\n%s/current_config.log\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), logsDir))

# Set iteration DataFrames depending on the arguments given by the user
# If NO custom file is provided

if args.file is False:

    if args.error is True:

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        cur_entries_df = pd.read_sql("SELECT * FROM %s WHERE processed = 2;" % ("entries",), pi_conn, index_col='pdb_id')

        proc_count = 0
        total_ent = len(cur_entries_df)

        pi_conn.close()

    elif args.current is True:

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        cur_entries_df = pd.read_sql("SELECT * FROM %s WHERE processed = 0;" % ("entries",), pi_conn, index_col='pdb_id')

        proc_count = 0
        total_ent = len(cur_entries_df)

        pi_conn.close()

    elif (args.new is False) and (args.restart is False):
        entries_df = pifaceFunctions.get_entries()

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        ex_entries_df = pd.read_sql("SELECT * FROM %s;" % "entries", pi_conn, index_col='pdb_id')
        cur_entries_df = pd.DataFrame([~entries_df.index.isin(ex_entries_df.index)])
        cur_entries_df['processed'] = 0
        cur_entries_df['update_time'] = update_time

        proc_count = 0
        total_ent = len(cur_entries_df)

        cur_entries_df.to_sql('entries', pi_conn, if_exists='append')

        pi_conn.close()

    elif (args.new is False) and (args.restart is True):

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        cur_entries_df = pd.read_sql("SELECT * FROM %s WHERE processed = 0 AND update_time = %s;" % ("entries", update_time), pi_conn, index_col='pdb_id')
        pi_cur.execute('SELECT COUNT(pdb_id) FROM entries WHERE update_time = ?', (update_time, ))
        total_ent = pi_cur.fetchone()[0]
        pi_cur.execute('SELECT COUNT(pdb_id) FROM entries WHERE update_time = ? AND processed = 1', (update_time, ))
        proc_count = pi_cur.fetchone()[0]

        pi_conn.close()

    elif args.new:
        entries_df = pifaceFunctions.get_entries()
        cur_entries_df = pd.DataFrame(entries_df)
        cur_entries_df['processed'] = 0
        cur_entries_df['update_time'] = update_time

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        proc_count = 0
        total_ent = len(cur_entries_df)

        cur_entries_df.to_sql('entries', pi_conn, if_exists='append')

        pi_conn.close()

# If custom file IS provided

if args.file:

    if not os.path.exists(args.file):
        raise SystemExit("Cannot find the file at path:\n%s" % args.file)

    custom_list = []

    with open(args.file, "r") as f:
        for line in f:
            custom_list = custom_list + [i.strip() for i in line.split(",")]

    if len(custom_list) == 0:
        raise SystemExit("No PDB IDs found in the file. Please check that your file is in a readable format, such as comma separated or single PDB ID in each line.")

    entries_df = pifaceFunctions.get_entries()

    if (args.new is False) and (args.restart is False):

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        ex_entries_df = pd.read_sql("SELECT * FROM %s;" % "entries", pi_conn, index_col='pdb_id')
        new_entries_df = entries_df[entries_df.index.isin(custom_list)]
        cur_entries_df = pd.DataFrame(new_entries_df[~new_entries_df.index.isin(ex_entries_df.index)])
        cur_entries_df['processed'] = 0
        cur_entries_df['update_time'] = update_time

        proc_count = 0
        total_ent = len(cur_entries_df)

        cur_entries_df.to_sql('entries', pi_conn, if_exists='append')

        pi_conn.commit()
        pi_conn.close()

    elif (args.new is False) and (args.restart is True):

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        cur_entries_df = pd.read_sql("SELECT * FROM %s WHERE processed = 0 AND update_time = %s;" % ("entries", update_time), pi_conn, index_col='pdb_id')
        pi_cur.execute('SELECT COUNT(pdb_id) FROM entries WHERE update_time = ?', (update_time, ))
        total_ent = pi_cur.fetchone()[0]
        pi_cur.execute('SELECT COUNT(pdb_id) FROM entries WHERE update_time = ? AND processed = 1', (update_time, ))
        proc_count = pi_cur.fetchone()[0]

        pi_conn.commit()
        pi_conn.close()

    elif args.new:

        cur_entries_df = pd.DataFrame(entries_df[entries_df.index.isin(custom_list)])
        cur_entries_df['processed'] = 0
        cur_entries_df['update_time'] = update_time

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        proc_count = 0
        total_ent = len(cur_entries_df)

        cur_entries_df.to_sql('entries', pi_conn, if_exists='append')

        pi_conn.commit()
        pi_conn.close()

cur_entries_df.to_csv("%s/currentEntries_%s.log" % (logsDir, update_time), sep=" ", mode="a")
execute_queries = []

for pdb_id, entry in cur_entries_df.iterrows():

    try:

        sql_queries = []

        proc_count += 1

        print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), " | ", pdb_id, " | ", proc_count, "/", total_ent)

        # Download the fasta file for the current pdb file and put it in the database.

        url = "https://www.rcsb.org/fasta/entry/%s" % pdb_id
        r = requests.get(url)
        open('%s.fasta' % pdb_id, 'wb').write(r.content)

        with open('%s.fasta' % pdb_id, 'r') as f:
            for line in f:
                if line.startswith('>'):
                    line_list = line.split('|')
                    pdb_id = line_list[0].split('_')[0][1:]
                    entity_id = line_list[0].split('_')[1]
                    chain_id = line_list[1].split(' ')[1]
                    info = line_list[2].strip()
                    source_organism = line_list[3].split('(')[0].strip()
                    # noinspection PyBroadException
                    try:
                        ncbi_taxonomy = int(line_list[3].split('(')[1].strip().rstrip(') '))

                    except:
                        source_organism = line_list[3].strip()
                        ncbi_taxonomy = "n/a"
                else:
                    sequence = line.strip('\n')
                    insert_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    query = 'INSERT INTO fasta (pdb_id, entity_id, chain_id, info, source_organism, ncbi_taxonomy, sequence, insert_time) VALUES(?,?,?,?,?,?,?,?)', (pdb_id, entity_id, chain_id, info, source_organism, ncbi_taxonomy, sequence, insert_time)

                    sql_queries.append(query)

        os.remove('%s.fasta' % pdb_id)

        # Download the cif file for the current pdb file and put it in the database.

        file_path = rcsb.fetch(pdb_id, "cif", os.getcwd())

        with open(pdb_id + '.cif', 'r') as cif_file:
            content = ''.join(cif_file.readlines())
            query = 'INSERT INTO cif (pdb_id, insert_time) VALUES(?,?)', (pdb_id, datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))

            sql_queries.append(query)

        # Check if there are dimers available in the cif file.
        # Write dimers and dimers_cif tables.

        cif_file = PDBxFile.read(file_path)
        structure = strucio.pdbx.get_structure(cif_file, model=1)
        model = structure[struc.filter_amino_acids(structure)]
        model = model[~model.hetero]

        os.remove('%s.cif' % pdb_id)

        chain_list = list(sorted(set(model.chain_id)))

        if len(chain_list) == 1:
            query = 'UPDATE cif SET chain_count = 1, combination_count = 0 WHERE pdb_id = ?', (pdb_id, )
            sql_queries.append(query)

        if len(chain_list) > 1:

            if len(chain_list) > 30:
                centroid = [pifaceFunctions.find_center(model[model.chain_id == ch].coord) for ch in chain_list]
                radius = [pifaceFunctions.find_R(model[model.chain_id == ch].coord) for ch in chain_list]

                dist = spatial.distance.cdist(np.array(centroid), np.array(centroid), 'euclidean')
                dist *= 1 - np.tri(*dist.shape, k=-1)

                dist_df = pd.DataFrame(data=dist, index=chain_list, columns=chain_list)
                near_df = dist_df[(dist_df < max(radius)) & (dist_df != 0)]

                chain_combinations = list(near_df.stack().index)

            else:
                chain_combinations = list(combinations(chain_list, 2))

            query = 'UPDATE cif SET chain_count = ?, combination_count = ? WHERE pdb_id = ?', (len(chain_list), len(chain_combinations), pdb_id)
            sql_queries.append(query)

            for comb in chain_combinations:
                dimer = model[np.logical_or(model.chain_id == comb[0], model.chain_id == comb[1])]
                sasa_result = pifaceFunctions.sasa(dimer, comb[0], comb[1])

                if sasa_result[0] + sasa_result[1] - sasa_result[2] > args.sasa:

                    pdbxFile = PDBxFile()
                    set_structure(pdbxFile, dimer, data_block=pdb_id+'_'+comb[0]+'_'+comb[1])
                    pdbxFile.write("%s_%s_%s.cif" % (pdb_id, comb[0], comb[1]))

                    dimer_id = pdb_id + '_' + comb[0] + '_'+ comb[1]

                    query = 'INSERT INTO dimers (dimer_id, pdb_id, chain_1, chain_2, chain_1_sasa, chain_2_sasa, dimer_sasa, sasa_diff, insert_time) VALUES(?,?,?,?,?,?,?,?,?)', (dimer_id, pdb_id, comb[0], comb[1], float(sasa_result[0]), float(sasa_result[1]), float(sasa_result[2]), float(sasa_result[0] + sasa_result[1] - sasa_result[2]), datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))
                    sql_queries.append(query)

                    with open("%s_%s_%s.cif" % (pdb_id, comb[0], comb[1]), 'r') as f:
                        content = ''.join(f.readlines())

                        query = 'INSERT INTO dimers_cif (dimer_id, cif_file, insert_time) VALUES(?,?,?)', (dimer_id, content, datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))
                        sql_queries.append(query)

                    os.remove("%s_%s_%s.cif" % (pdb_id, comb[0], comb[1]))

                    # Check if there is an interface in the dimer.

                    # Write interfaces, interfaces_cif and interfaces_seq tables.

                    extInterface = pifaceFunctions.extractInterface(dimer, comb[0], comb[1], args.vdw, args.mincontacting, args.nearby, args.memory)

                    if extInterface is not None:

                        query = 'INSERT INTO interfaces (dimer_id, pdb_id, chain_1, chain_2, chain_1_contacting, chain_2_contacting, chain_1_nearby, chain_2_nearby, insert_time) VALUES(?,?,?,?,?,?,?,?,?)', (dimer_id, pdb_id, comb[0], comb[1], extInterface[0], extInterface[1], extInterface[2], extInterface[3], datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))
                        sql_queries.append(query)

                        output = PDBxFile()
                        set_structure(output, extInterface[4], data_block=dimer_id)
                        output.write("%s.cif" % dimer_id)

                        with open("%s.cif" % dimer_id, 'r') as f:
                            content = ''.join(f.readlines())

                            query = 'INSERT INTO interfaces_cif (dimer_id, cif_file, insert_time) VALUES(?,?,?)', (dimer_id, content, datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))
                            sql_queries.append(query)

                        os.remove("%s.cif" % dimer_id)

                        seq_1 = ", ".join(map(str, list(struc.get_residues(extInterface[4][extInterface[4].chain_id == comb[0]])[1])))
                        seq_2 = ", ".join(map(str, list(struc.get_residues(extInterface[4][extInterface[4].chain_id == comb[1]])[1])))

                        query = 'INSERT INTO interfaces_seq (dimer_id, pdb_id, chain_1, chain_2, sequence_1, sequence_2, insert_time) VALUES(?,?,?,?,?,?,?)', (dimer_id, pdb_id, comb[0], comb[1], seq_1, seq_2, datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S"))
                        sql_queries.append(query)

        query = 'UPDATE entries SET processed = 1 WHERE pdb_id = ?', (pdb_id, )
        sql_queries.append(query)

        # Add queries to another list, to execute together later. It is to ensure that no table is affected from the changes, and also to keep the database closed during the runs.

        execute_queries = execute_queries + sql_queries

    except Exception as e:

        with open("%s/errors_%s.log" % (logsDir, update_time), "a") as err_file:
            print("Following PDB ID has raised an error: %s\n" % pdb_id, file=err_file)
            print(sys.exc_info()[0], file=err_file)
            print(sys.exc_info()[1], file=err_file)
            print(traceback.format_exc(), file=err_file)
            print("\n- - - - - - - - - -\n", file=err_file)

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        pi_cur.execute('UPDATE entries SET processed = 2 WHERE pdb_id = ?', (pdb_id, ))

        pi_conn.commit()
        pi_conn.close()

        continue

    if proc_count % 100 == 0:

        pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
        pi_cur = pi_conn.cursor()

        for query in execute_queries:
            pi_cur.execute(query[0], query[1])

        execute_queries = []

        pi_conn.commit()
        pi_conn.close()

os.rename('%s/current_config.log' % logsDir, '%s/config_%s.log' % (logsDir, update_time))

if len(execute_queries) > 0:

    pi_conn = sqlite3.connect("%s/%s" % (workDir, args.database))
    pi_cur = pi_conn.cursor()

    for query in execute_queries:
        pi_cur.execute(query[0], query[1])

    execute_queries = []

    pi_conn.commit()
    pi_conn.close()
