import sqlite3
import pandas as pd
import os
import glob
import asyncio
import aiosqlite
import aiofiles
import time
from aiofiles import os as aios
from sqlalchemy import create_engine, and_, update, Column, Integer, MetaData, String, Boolean, Table, DateTime, Float
from sqlalchemy.ext.asyncio import create_async_engine
from datetime import datetime
import subprocess
import newFunctions


async def read_results(file_name):

    mmalign_file = file_name
    trans_file = file_name.replace("mmalign", "trans")

    int_1 = "_".join(mmalign_file.split(".")[0].split("_")[0:3])
    int_2 = "_".join(mmalign_file.split(".")[0].split("_")[3:6])

    async with aiofiles.open(mmalign_file, 'r') as mmalign_result:
        result_list = await mmalign_result.readlines()

    file_1 = result_list[7].split(":")[1].strip()
    file_2 = result_list[8].split(":")[1].strip()
    al = int(result_list[12].split(",")[0].split("=")[1])
    rd = float(result_list[12].split(",")[1].split("=")[1])
    si = float(result_list[12].split(",")[2].split("=")[2])
    tm_1 = float(result_list[13].split("=")[1].split("(")[0])
    tm_2 = float(result_list[14].split("=")[1].split("(")[0])
    ar = ''.join(result_list[18:21])

    async with aiofiles.open(trans_file, 'r') as trans_result:
        trans_list = await trans_result.readlines()

    data = [list(filter(None, i.strip().split(" "))) for i in trans_list if i[0] in ["0", "1", "2"]]

    t0_v, t1_v, t2_v = float(data[0][1]), float(data[1][1]), float(data[2][1])
    u00_v, u01_v, u02_v = float(data[0][2]), float(data[0][3]), float(data[0][4])
    u10_v, u11_v, u12_v = float(data[1][2]), float(data[1][3]), float(data[1][4])
    u20_v, u21_v, u22_v = float(data[2][2]), float(data[2][3]), float(data[2][4])

    if rd <= 1:
        st = 1
    else:
        st = 0

    async with async_engine.begin() as conn:
        await conn.execute(update(comparison_seq).where(
            and_(comparison_seq.c.interface_1 == int_1, comparison_seq.c.interface_2 == int_2)).values(
            aligned_length=al, rmsd=rd, sequence_identity=si, tmscore_int1=tm_1, tmscore_int2=tm_2, aligned_residues=ar,
            t0=t0_v, t1=t1_v, t2=t2_v, u00=u00_v, u01=u01_v, u02=u02_v, u10=u01_v, u11=u11_v, u12=u12_v, u20=u20_v,
            u21=u21_v, u22=u22_v, processed=1, status=st, insert_time=datetime.now()))

    await aios.remove(file_1)
    await aios.remove(file_2)
    await aios.remove(mmalign_file)
    await aios.remove(trans_file)


async def main():

    cor = [read_results(file_name) for file_name in current_files]

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
    async_engine = create_async_engine('mysql+mysqldb://dbConnection/interfaceDB')

    while True:
        current_files = glob.glob("*.mmalign")

        if len(current_files) > 1000:
            time.sleep(3)
            asyncio.run(main())

        else:
            continue
