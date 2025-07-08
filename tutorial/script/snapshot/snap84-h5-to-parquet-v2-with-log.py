#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on 06 2025


output paths are hard-wired 
    hdfsheader = 'hdfs://spark00:54310'
    hdfsgaspath = '/common/data/illustris/tng50/snapshot84/gas/'
    hdfsdmpath = '/common/data/illustris/tng50/snapshot84/dm/'
    hdfsstarpath = '/common/data/illustris/tng50/snapshot84/star/'
    hdfsbhpath = '/common/data/illustris/tng50/snapshot84/bh/'


@author: shong
"""

import numpy as np
import pandas as pd
import glob
import sys
import h5py
#from netCDF4 import Dataset
import matplotlib.pyplot as plt

import pyarrow as pa
import pyarrow.parquet as pq

from functools import reduce
import operator
import gc


# PySpark packages
from pyspark import SparkContext   
from pyspark.sql import SparkSession

import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark import Row
from pyspark.sql.window import Window as W

import logging
from pathlib import Path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage : <command> <infile_hdf5> [logfile_path]")
        sys.exit()
    else:
        infile = sys.argv[1]
        logfile = sys.argv[2] if len(sys.argv) > 2 else "parquet_conversion_master.log"
        
    # Ensure logfile exists
    Path(logfile).touch(exist_ok=True)
    logging.basicConfig(
        filename=logfile,
        filemode='a',  # append mode
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"=== Starting conversion for: {infile} ===")


    # Define Spark Session
    print("## Logging File = "+logfile)
    print("## Program Log : Defining SparkSession...")
    
    spark = SparkSession.builder \
        .master("yarn") \
        .appName("spark-shell") \
        .config("spark.driver.maxResultSize", "32g") \
        .config("spark.driver.memory", "32g") \
        .config("spark.executor.memory", "7g") \
        .config("spark.executor.cores", "1") \
        .config("spark.executor.instances", "200") \
        .getOrCreate()
    
    sc = spark.sparkContext
    sc.setCheckpointDir("hdfs://spark00:54310/tmp/checkpoints")

    spark.conf.set("spark.sql.debug.maxToStringFields", 500)
    spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")



    # infile and outfile names from sys.argv
    infile  = sys.argv[1]
    #outfile = sys.argv[2]
    
    hdfsheader = 'hdfs://spark00:54310'
    hdfsgaspath = '/common/data/illustris/tng50/snapshot84/gas/'
    hdfsdmpath = '/common/data/illustris/tng50/snapshot84/dm/'
    hdfsstarpath = '/common/data/illustris/tng50/snapshot84/star/'
    hdfsbhpath = '/common/data/illustris/tng50/snapshot84/bh/'
    
    parsedname = infile.split("/")[-1].replace(".hdf5", "")

    
    ############## Read h5 ###############
    try:
        h5f = h5py.File(infile, "r")
    except IOError as e:
        print("Error opening HDF5 file:", str(e))
    # Don't forget f.close() when done! 


    gas_schema = T.StructType([
        T.StructField("CenterOfMass", T.ArrayType(T.FloatType()), True),
        T.StructField("Coordinates", T.ArrayType(T.DoubleType()), True),
        T.StructField("Density", T.FloatType(), True),
        T.StructField("ElectronAbundance", T.FloatType(), True),
        T.StructField("EnergyDissipation", T.FloatType(), True),
        T.StructField("GFM_AGNRadiation", T.FloatType(), True),
        T.StructField("GFM_CoolingRate", T.FloatType(), True),
        T.StructField("GFM_Metallicity", T.FloatType(), True),
        T.StructField("GFM_Metals", T.ArrayType(T.FloatType()), True),
        T.StructField("GFM_MetalsTagged", T.ArrayType(T.FloatType()), True),
        T.StructField("GFM_WindDMVelDisp", T.FloatType(), True),
        T.StructField("GFM_WindHostHaloMass", T.FloatType(), True),
        T.StructField("InternalEnergy", T.FloatType(), True),
        T.StructField("InternalEnergyOld", T.FloatType(), True),
        T.StructField("Machnumber", T.FloatType(), True),
        T.StructField("MagneticField", T.ArrayType(T.FloatType()), True),
        T.StructField("MagneticFieldDivergence", T.FloatType(), True),
        T.StructField("Masses", T.FloatType(), True),
        T.StructField("NeutralHydrogenAbundance", T.FloatType(), True),
        T.StructField("ParticleIDs", T.LongType(), True),
        T.StructField("Potential", T.FloatType(), True),
        T.StructField("StarFormationRate", T.FloatType(), True),
        T.StructField("SubfindDMDensity", T.FloatType(), True),
        T.StructField("SubfindDensity", T.FloatType(), True),
        T.StructField("SubfindHsml", T.FloatType(), True),
        T.StructField("SubfindVelDisp", T.FloatType(), True),
        T.StructField("Velocities", T.ArrayType(T.FloatType()), True)])
    
    try:
        logging.info("Processing GAS particles...")
        gas_group = h5f['PartType0']
        gas_data = {key: gas_group[key][:].tolist() for key in gas_group.keys()}
        gas_data_tuples = list(zip(*gas_data.values()))
        outgasname = hdfsheader + hdfsgaspath + parsedname + '_gas.parquet.snappy'

        spark.createDataFrame(gas_data_tuples, schema=gas_schema) \
            .write.option("compression", "snappy") \
            .mode("overwrite") \
            .save(outgasname)

        logging.info(f"✅ Successfully saved GAS parquet: {outgasname}")

    except Exception as e:
        logging.error(f"❌ Failed to save GAS parquet: {outgasname}. Error: {str(e)}")


    dm_schema = T.StructType([
        T.StructField("Coordinates", T.ArrayType(T.DoubleType()), True),
        T.StructField("ParticleIDs", T.LongType(), True),
        T.StructField("Potential", T.FloatType(), True),
        T.StructField("SubfindDMDensity", T.FloatType(), True),
        T.StructField("SubfindDensity", T.FloatType(), True),
        T.StructField("SubfindHsml", T.FloatType(), True),
        T.StructField("SubfindVelDisp", T.FloatType(), True),
        T.StructField("Velocities", T.ArrayType(T.FloatType()), True),
    ])
    
    try:
        logging.info("Processing DM particles...")
        dm_group = h5f['PartType1']
        dm_data = {key: dm_group[key][:].tolist() for key in dm_group.keys()}
        dm_data_tuples = list(zip(*dm_data.values()))
        outdmname = hdfsheader + hdfsdmpath + parsedname + '_dm.parquet.snappy'

        spark.createDataFrame(dm_data_tuples, schema=dm_schema) \
            .write.option("compression", "snappy") \
            .mode("overwrite") \
            .save(outdmname)

        logging.info(f"✅ Successfully saved DM parquet: {outdmname}")

    except Exception as e:
        logging.error(f"❌ Failed to save DM parquet: {outdmname}. Error: {str(e)}")
    
    
    
    star_schema = T.StructType([
        T.StructField("BirthPos", T.ArrayType(T.FloatType()), True),
        T.StructField("BirthVel", T.ArrayType(T.FloatType()), True),
        T.StructField("Coordinates", T.ArrayType(T.DoubleType()), True),
        T.StructField("GFM_InitialMass", T.FloatType(), True),
        T.StructField("GFM_Metallicity", T.FloatType(), True),
        T.StructField("GFM_Metals", T.ArrayType(T.FloatType()), True),
        T.StructField("GFM_MetalsTagged", T.ArrayType(T.FloatType()), True),
        T.StructField("GFM_StellarFormationTime", T.FloatType(), True),
        T.StructField("GFM_StellarPhotometrics", T.ArrayType(T.FloatType()), True),
        T.StructField("Masses", T.FloatType(), True),
        T.StructField("ParticleIDs", T.LongType(), True),
        T.StructField("Potential", T.FloatType(), True),
        T.StructField("StellarHsml", T.FloatType(), True),
        T.StructField("SubfindDMDensity", T.FloatType(), True),
        T.StructField("SubfindDensity", T.FloatType(), True),
        T.StructField("SubfindHsml", T.FloatType(), True),
        T.StructField("SubfindVelDisp", T.FloatType(), True),
        T.StructField("Velocities", T.ArrayType(T.FloatType()), True),
    ])
    try:
        logging.info("Processing STAR particles...")
        star_group = h5f['PartType4']
        star_data = {key: star_group[key][:].tolist() for key in star_group.keys()}
        star_data_tuples = list(zip(*star_data.values()))
        outstarname = hdfsheader + hdfsstarpath + parsedname + '_star.parquet.snappy'

        spark.createDataFrame(star_data_tuples, schema=star_schema) \
            .write.option("compression", "snappy") \
            .mode("overwrite") \
            .save(outstarname)

        logging.info(f"✅ Successfully saved STAR parquet: {outstarname}")

    except Exception as e:
        logging.error(f"❌ Failed to save STAR parquet: {outstarname}. Error: {str(e)}")

    
    bh_schema = T.StructType([
        T.StructField("BH_BPressure", T.FloatType(), True),
        T.StructField("BH_CumEgyInjection_QM", T.FloatType(), True),
        T.StructField("BH_CumEgyInjection_RM", T.FloatType(), True),
        T.StructField("BH_CumMassGrowth_QM", T.FloatType(), True),
        T.StructField("BH_CumMassGrowth_RM", T.FloatType(), True),
        T.StructField("BH_Density", T.FloatType(), True),
        T.StructField("BH_HostHaloMass", T.FloatType(), True),
        T.StructField("BH_Hsml", T.FloatType(), True),
        T.StructField("BH_Mass", T.FloatType(), True),
        T.StructField("BH_Mdot", T.FloatType(), True),
        T.StructField("BH_MdotBondi", T.FloatType(), True),
        T.StructField("BH_MdotEddington", T.FloatType(), True),
        T.StructField("BH_Pressure", T.FloatType(), True),
        T.StructField("BH_Progs", T.IntegerType(), True),  # uint32 → safe mapping to IntegerType
        T.StructField("BH_U", T.FloatType(), True),
        T.StructField("Coordinates", T.ArrayType(T.DoubleType()), True),
        T.StructField("Masses", T.FloatType(), True),
        T.StructField("ParticleIDs", T.LongType(), True),
        T.StructField("Potential", T.FloatType(), True),
        T.StructField("SubfindDMDensity", T.FloatType(), True),
        T.StructField("SubfindDensity", T.FloatType(), True),
        T.StructField("SubfindHsml", T.FloatType(), True),
        T.StructField("SubfindVelDisp", T.FloatType(), True),
        T.StructField("Velocities", T.ArrayType(T.FloatType()), True),
    ])
    
    try:
        logging.info("Processing BH particles...")
        bh_group = h5f['PartType5']
        bh_data = {key: bh_group[key][:].tolist() for key in bh_group.keys()}
        bh_data_tuples = list(zip(*bh_data.values()))
        outbhname = hdfsheader + hdfsbhpath + parsedname + '_bh.parquet.snappy'

        spark.createDataFrame(bh_data_tuples, schema=bh_schema) \
            .write.option("compression", "snappy") \
            .mode("overwrite") \
            .save(outbhname)

        logging.info(f"✅ Successfully saved BH parquet: {outbhname}")

    except Exception as e:
        logging.error(f"❌ Failed to save BH parquet: {outbhname}. Error: {str(e)}")
    
    h5f.close()
    sc.stop()