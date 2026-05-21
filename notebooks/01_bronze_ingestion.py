# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Bronze Ingestion
# MAGIC
# MAGIC **What is Bronze?**
# MAGIC Bronze is the first layer of the Medallion Architecture.
# MAGIC We read raw files EXACTLY as they are from ADLS Gen2 and write
# MAGIC them to Delta Tables. No cleaning, no transformations. Just land the data.
# MAGIC Think of it as taking a photograph of the original data before we touch anything.

# COMMAND ----------

# ── STORAGE CONFIGURATION ──────────────────────────────────────────────────
# Before Spark can read from Azure Data Lake Storage (ADLS Gen2),
# it needs to know HOW to authenticate. We're using OAuth which is
# the secure Microsoft standard for service-to-service authentication.
# Think of this like giving Spark a keycard to enter the storage building.

storage_account = "matchiqstorage"  # The name of our ADLS Gen2 account
container = "matchiq"               # The container (top-level folder) inside it

# Tell Spark to use OAuth authentication for our storage account
# This is like telling Spark "use this type of ID badge"
spark.conf.set(
    f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net",
    "OAuth"
)

# Tell Spark which OAuth provider to use
# ClientCredsTokenProvider = authenticate using client credentials (app ID + secret)
spark.conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
)

# COMMAND ----------

# ── BASE PATH ──────────────────────────────────────────────────────────────
# abfss:// is the Azure Blob File System Secure protocol
# This is the address format ADLS Gen2 uses — like a URL but for data lakes
# Think of it as the root address of our entire data lake
# Everything we read or write will start with this path

base_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest Speed Dating Dataset
# MAGIC
# MAGIC Reading the raw CSV from ADLS Gen2 into a Spark DataFrame.
# MAGIC A DataFrame in Spark is like a pandas DataFrame but distributed —
# MAGIC instead of living on your laptop it lives across all worker nodes simultaneously.

# COMMAND ----------

# ── READ SPEED DATING CSV ──────────────────────────────────────────────────
# spark.read is how we tell Spark to load data
# .format("csv") tells Spark what file type it is
# .option("header", "true") means the first row contains column names
# .option("inferSchema", "true") tells Spark to guess data types automatically
#   (e.g. this column looks like integers, this one looks like strings)
# .option("encoding", "UTF-8") handles special characters in the text
# .load() is where we point Spark at the actual file path

speed_dating_raw = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .option("encoding", "UTF-8") \
    .load(f"{base_path}/raw/speed_dating/speed_dating.csv")

# COMMAND ----------

# ── QUICK SANITY CHECK ─────────────────────────────────────────────────────
# Before writing anything we always verify the data loaded correctly
# .count() triggers an actual Spark job to count all rows across all partitions
# .columns gives us the list of column names
# .printSchema() shows us every column name + data type Spark inferred

print(f"Speed Dating rows: {speed_dating_raw.count()}")          # Expect ~8,378
print(f"Speed Dating columns: {len(speed_dating_raw.columns)}")  # Expect ~195
speed_dating_raw.printSchema()                                    # Shows column types

# COMMAND ----------

# ── WRITE TO BRONZE DELTA TABLE ────────────────────────────────────────────
# Now we write the raw data to a Delta Table in our Bronze layer
# .format("delta") = write as a Delta Table (not plain parquet or csv)
#   Delta adds a transaction log on top of parquet files giving us:
#   - ACID transactions (writes either fully succeed or fully fail)
#   - Time travel (query the table as it looked at any previous point)
#   - Schema enforcement (rejects bad data types automatically)
# .mode("overwrite") = if a Bronze table already exists, replace it
#   This makes our pipeline idempotent — safe to re-run without duplicating data
# .save() = write to this path in ADLS Gen2

speed_dating_raw.write \
    .format("delta") \
    .mode("overwrite") \
    .save(f"{base_path}/bronze/speed_dating")

print("✓ Speed Dating Bronze layer written successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ingest OKCupid Dataset
# MAGIC
# MAGIC OKCupid comes as a Parquet file, not CSV.
# MAGIC Parquet is a columnar storage format — instead of storing data row by row
# MAGIC it stores column by column. This makes it much faster for analytics
# MAGIC because Spark only reads the columns it needs instead of every row.
# MAGIC Think of CSV as a spreadsheet and Parquet as a database optimized for reading.

# COMMAND ----------

# ── READ OKCUPID PARQUET ───────────────────────────────────────────────────
# Parquet already knows its own schema so we don't need inferSchema
# Spark reads the schema directly from the file metadata
# This is one reason Parquet is better than CSV for data engineering

okcupid_raw = spark.read.format("parquet") \
    .load(f"{base_path}/raw/okcupid/parsed_data_public.parquet")

# COMMAND ----------

# ── QUICK SANITY CHECK ─────────────────────────────────────────────────────
print(f"OKCupid rows: {okcupid_raw.count()}")           # Expect ~68,371
print(f"OKCupid columns: {len(okcupid_raw.columns)}")   # Expect ~2,620
okcupid_raw.printSchema()                               # Shows all column types

# COMMAND ----------

# ── WRITE TO BRONZE DELTA TABLE ────────────────────────────────────────────
# Same pattern as speed dating — land raw data into Bronze as a Delta Table
# We never modify raw data. If something breaks downstream,
# we can always come back to Bronze and start over cleanly.

okcupid_raw.write \
    .format("delta") \
    .mode("overwrite") \
    .save(f"{base_path}/bronze/okcupid")

print("✓ OKCupid Bronze layer written successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Bronze Tables
# MAGIC
# MAGIC Always verify after writing. Read the Delta Tables back and confirm
# MAGIC the row counts match what we loaded. This is a basic data quality check —
# MAGIC if rows are missing something went wrong during the write.

# COMMAND ----------

# ── READ BACK AND VERIFY ───────────────────────────────────────────────────
# We read the Delta Tables back from ADLS to confirm they wrote correctly
# If these counts match what we saw above, Bronze ingestion is complete

speed_bronze = spark.read.format("delta") \
    .load(f"{base_path}/bronze/speed_dating")

okcupid_bronze = spark.read.format("delta") \
    .load(f"{base_path}/bronze/okcupid")

print(f"✓ Bronze Speed Dating: {speed_bronze.count()} rows")
print(f"✓ Bronze OKCupid: {okcupid_bronze.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Bronze Ingestion Complete
# MAGIC
# MAGIC Both datasets are now in Bronze Delta Tables in ADLS Gen2.
# MAGIC Next step: 02_silver_cleaning.py
# MAGIC We will clean, deduplicate, fix data types, and handle missing values.