#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import sqlite3
from datetime import datetime
import time
from itertools import combinations
from sqlalchemy import create_engine, Column, Integer, MetaData, String, Boolean, Table, DateTime, Float, Double
from sqlalchemy.ext.asyncio import create_async_engine


conn = sqlite3.connect('/kuacc/users/zabali16/NEW_PIFace/piface.db')
cur = conn.cursor()

interface_df = pd.read_sql_query('SELECT * FROM interfaces', conn)
interface_seq_df = pd.read_sql_query('SELECT * FROM interfaces_seq', conn)

conn.close()


interface_seq_df['forward'] = interface_seq_df['sequence_1'] + interface_seq_df['sequence_2']
interface_seq_df['backward'] = interface_seq_df['sequence_2'] + interface_seq_df['sequence_1']

forward_seq = interface_seq_df.groupby('forward')['dimer_id'].apply(list)
backward_seq = interface_seq_df.groupby('backward')['dimer_id'].apply(list)

all_seq = list(forward_seq) + list(backward_seq)

same_seq = [i for i in all_seq if len(i) > 1]

ind_seq = list(set([val for sublist in same_seq for val in sublist]))


total = len(ind_seq)
all_groups = []
count = 0

print(f"#{datetime.now()} Starting sequential identitiy comparison for {len(interface_df)} interfaces.")

start = datetime.now()

for i in ind_seq:
    count += 1
    temp = [group for group in all_seq if i in group]
    group = list(set([val for sublist in temp for val in sublist]))
    
    all_groups.append(group)
    
    now = datetime.now()
    
    if count % 100 == 0:
        print(f"Progress: {count}/{total} Time: {now - start}")
        
unique_all_groups = []

count = 0
total = len(all_groups)

for i in all_groups:
    
    count += 1
    
    if i not in unique_all_groups:
        unique_all_groups.append(i)
    else:
        continue

print(f"#{datetime.now()} Setting up the database.")

comp_conn = sqlite3.connect('comparisons.db')
comp_cur = comp_conn.cursor()

comp_cur.execute(""" PRAGMA journal_mode=WAL; """)

comp_cur.execute("""CREATE TABLE IF NOT EXISTS interfaces (
                    dimer_id text NOT NULL,
                    processed integer NOT NULL,
                    update_date text NOT NULL
                    )""")
comp_cur.execute(""" CREATE INDEX idx_interfaces ON interfaces(dimer_id) """)

comp_cur.execute("""CREATE TABLE IF NOT EXISTS identical_seq (
                    dimer_id text NOT NULL,
                    group_id integer NOT NULL,
                    insert_time text NOT NULL    
                    )""")
comp_cur.execute(""" CREATE INDEX idx_identical_seq ON identical_seq(dimer_id) """)

comp_conn.close()


print(f"{datetime.now()} Writing identical groups and comparisons to database.")

comp_conn = sqlite3.connect('comparisons.db')
comp_cur = comp_conn.cursor()

interface_sql = pd.DataFrame(interface_df.dimer_id)
interface_sql['processed'] = 1
interface_sql['update_date'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
interface_sql.set_index('dimer_id', inplace=True)

conn = sqlite3.connect('comparisons.db')

interface_sql.to_sql('interfaces', comp_conn, if_exists='append')

group_id = 0

for group in unique_all_groups:
    for member in group:
        comp_cur.execute('INSERT INTO identical_seq (dimer_id, group_id, representative, processed, insert_time) VALUES(?,?,?,?,?)', (member, group_id, 0, 0, datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")))
        
    group_id += 1
    
    if group_id % 100 == 0:
        comp_conn.commit()
    
groups_df = pd.read_sql_query('SELECT * from identical_seq', comp_conn)

comp_conn.commit()
comp_conn.close()

all_groups = groups_df.groupby('group_id')['dimer_id'].apply(list)

engine = create_engine('mysql+mysqldb://dbConnection/interfaceDB')
metadata = MetaData()

comparison_seq = Table(
    'comparison_seq', metadata,
    Column('interface_1', String(16, collation="latin1_general_cs"), nullable=False, primary_key=True),
    Column('interface_2', String(16, collation="latin1_general_cs"), nullable=False, primary_key=True),
    Column('group_id', Integer, nullable=False),
    Column('aligned_length', Integer),
    Column('rmsd', Float),
    Column('sequence_identity', Float),
    Column('tmscore_int1', Float),
    Column('tmscore_int2', Float),
    Column('aligned_residues', Text),
    Column('t0', Double),
    Column('t1', Double),
    Column('t2', Double),
    Column('u00', Double),
    Column('u01', Double),
    Column('u02', Double),
    Column('u10', Double),
    Column('u11', Double),
    Column('u12', Double),
    Column('u20', Double),
    Column('u21', Double),
    Column('u22', Double),
    Column('processed', Integer, nullable=False),
    Column('status', Integer, nullable=False),
    Column('insert_time', DateTime)
)

comparisons = Table(
    'comparisons', metadata,
    Column('interface_1', String(16, collation="latin1_general_cs"), nullable=False, primary_key=True),
    Column('interface_2', String(16, collation="latin1_general_cs"), nullable=False, primary_key=True),
    Column('group_id', Integer),
    Column('aligned_length', Integer),
    Column('rmsd', Float),
    Column('sequence_identity', Float),
    Column('tmscore_int1', Float),
    Column('tmscore_int2', Float),
    Column('aligned_residues', Text),
    Column('t0', Double),
    Column('t1', Double),
    Column('t2', Double),
    Column('u00', Double),
    Column('u01', Double),
    Column('u02', Double),
    Column('u10', Double),
    Column('u11', Double),
    Column('u12', Double),
    Column('u20', Double),
    Column('u21', Double),
    Column('u22', Double),
    Column('processed', Integer, nullable=False),
    Column('status', Integer, nullable=False),
    Column('insert_time', DateTime)
)

metadata.create_all(engine)

count = 0
total = len(all_groups)

print(f"{datetime.now()} Writing comparisons to MySql DB.")

for index, group in all_groups.iteritems():
    count += 1
    
    group_combs = combinations(group, 2)
    
    for comb in group_combs:
        with engine.connect() as conn:
            conn.execute(comparison_seq.insert(), [{"interface_1": comb[0], "interface_2": comb[1], "group_id": index, "processed": 0, "status": 0, "insert_time": datetime.now()}])
