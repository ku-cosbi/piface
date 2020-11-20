#!/usr/bin/env python
# coding: utf-8

import sqlite3
import pandas as pd
import os
import asyncio
import aiosqlite
import aiofiles
from aiofiles import os as aios
from sqlalchemy import create_engine, and_, update, Column, Integer, MetaData, String, Boolean, Table, DateTime, Float
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime
import subprocess
import newFunctions


async def read_sql(comb):

    async with aiosqlite.connect("/kuacc/users/zabali16/NEW_PIFace/piface.db") as db:

        file_1 = f"{newFunctions.get_random_string(14)}.cif"
        file_2 = f"{newFunctions.get_random_string(14)}.cif"

        async with db.execute("SELECT cif_file from interfaces_cif WHERE dimer_id = ?;", (comb[0], )) as cur:
            async for row in cur:
                async with aiofiles.open(file_1, "a") as f:

                    data = [''.join([i.strip(), " ", '\n']) if i.startswith("_atom_site") else ''.join([i.strip(), '\n']) for i in row[0].split("\n")]

                    await f.write("".join(data))
        
        async with db.execute("SELECT cif_file from interfaces_cif WHERE dimer_id = ?;", (comb[1], )) as cur:
            async for row in cur:
                async with aiofiles.open(file_2, "a") as f:

                    data = [''.join([i.strip(), " ", '\n']) if i.startswith("_atom_site") else ''.join([i.strip(), '\n']) for i in row[0].split("\n")]

                    await f.write("".join(data))

    await asyncio.create_subprocess_shell(f"MMalign {file_1} {file_2} -m {comb[0]}_{comb[1]}.trans > {comb[0]}_{comb[1]}.mmalign\n")


async def main():
    
    cor = [read_sql(comb) for comb in combs]
    
    no_concurrent = 100
    tasks = set()
    i = 0
    
    for i in range(len(cor)):
        if len(tasks) >= no_concurrent:
            _done, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        tasks.add(asyncio.create_task(cor[i]))
        i += 1

    await asyncio.wait(tasks)

if __name__ == '__main__':

    engine = create_engine('mysql+mysqldb://dbConnection/interfaceDB')
    metadata = MetaData()
    comparison_seq = Table('comparison_seq', metadata, autoload=True, autoload_with=engine)

    with engine.connect() as conn:
        comparisons_df = pd.read_sql_query('SELECT interface_1, interface_2 FROM comparison_seq WHERE processed=0', conn, chunksize=10000)

    for chunk in comparisons_df:

        combs = list(zip(chunk.interface_1, chunk.interface_2))

        asyncio.run(main())

    conn.close()
