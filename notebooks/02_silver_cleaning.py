# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 - Silver Cleaning
# MAGIC
# MAGIC **What is Silver?**
# MAGIC Silver is the second layer of the Medallion Architecture.
# MAGIC We take the raw Bronze data and make it trustworthy.
# MAGIC This means fixing data types, handling missing values,
# MAGIC removing duplicates, and renaming confusing columns.
# MAGIC Think of it as taking the raw crime scene photo and organizing
# MAGIC it into a clean, readable case file.

# COMMAND ----------

# ── IMPORTS ────────────────────────────────────────────────────────────────
# We import specific PySpark functions we need for cleaning
# These are like Excel formulas but they run across millions of rows in parallel
# col() = reference a column by name
# when() = if/else logic (like Excel IF)
# isnan() = check if a value is NaN (Not a Number)
# isnull() = check if a value is None/null
# mean() = calculate average (used for imputation)
# regexp_replace() = find and replace using patterns (like regex in Python)
# trim() = remove whitespace from strings

from pyspark.sql.functions import (
    col, when, isnan, isnull, mean,
    regexp_replace, trim, lower,
    count, round, lit
)
from pyspark.sql.types import IntegerType, FloatType, StringType

# COMMAND ----------

# ── STORAGE CONFIGURATION ──────────────────────────────────────────────────
# Same storage config as Bronze — we need to authenticate to ADLS Gen2
# every time we connect to it in a new notebook

storage_account = "matchiqstorage"
container = "matchiq"

spark.conf.set(
    f"fs.azure.account.auth.type.{storage_account}.dfs.core.windows.net",
    "OAuth"
)
spark.conf.set(
    f"fs.azure.account.oauth.provider.type.{storage_account}.dfs.core.windows.net",
    "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider"
)

base_path = f"abfss://{container}@{storage_account}.dfs.core.windows.net"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Bronze Tables
# MAGIC
# MAGIC We always read FROM the previous layer, never from raw.
# MAGIC Bronze is our source of truth for Silver.

# COMMAND ----------

# ── READ FROM BRONZE ───────────────────────────────────────────────────────
# Reading Delta Tables is fast because Delta stores metadata about
# where each partition lives — Spark doesn't have to scan the whole lake

speed_df = spark.read.format("delta") \
    .load(f"{base_path}/bronze/speed_dating")

okcupid_df = spark.read.format("delta") \
    .load(f"{base_path}/bronze/okcupid")

print(f"Bronze Speed Dating loaded: {speed_df.count()} rows")
print(f"Bronze OKCupid loaded: {okcupid_df.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean Speed Dating Dataset
# MAGIC
# MAGIC The speed dating dataset has 195 columns and real-world messiness:
# MAGIC - Missing ratings (people skipped questions)
# MAGIC - Inconsistent data types (numbers stored as strings)
# MAGIC - Duplicate encounter rows
# MAGIC - Columns with cryptic names like iid, pid, pf_o_att

# COMMAND ----------

# ── STEP 1: DEDUPLICATE ────────────────────────────────────────────────────
# Each row should represent ONE unique encounter between two people
# iid = person's unique ID, pid = their partner's unique ID
# If the same pair appears twice, we keep only one row
# .dropDuplicates() removes exact duplicate rows
# We pass the key columns that define a unique encounter

speed_df = speed_df.dropDuplicates(["iid", "pid"])
print(f"After dedup: {speed_df.count()} rows")

# COMMAND ----------

# ── STEP 2: FIX DATA TYPES ─────────────────────────────────────────────────
# inferSchema in Bronze made some guesses that might be wrong
# For example age might have been read as string if it had any bad values
# We explicitly cast the columns we know should be numeric
# .cast() converts a column to the specified type
# If a value can't be converted (e.g. "N/A" → integer), it becomes null
# which we'll handle in the imputation step below

speed_df = speed_df \
    .withColumn("age", col("age").cast(FloatType())) \
    .withColumn("age_o", col("age_o").cast(FloatType())) \
    .withColumn("match", col("match").cast(IntegerType())) \
    .withColumn("attr", col("attr").cast(FloatType())) \
    .withColumn("sinc", col("sinc").cast(FloatType())) \
    .withColumn("intel", col("intel").cast(FloatType())) \
    .withColumn("fun", col("fun").cast(FloatType())) \
    .withColumn("amb", col("amb").cast(FloatType())) \
    .withColumn("shar", col("shar").cast(FloatType())) \
    .withColumn("attr_o", col("attr_o").cast(FloatType())) \
    .withColumn("sinc_o", col("sinc_o").cast(FloatType())) \
    .withColumn("intel_o", col("intel_o").cast(FloatType())) \
    .withColumn("fun_o", col("fun_o").cast(FloatType())) \
    .withColumn("amb_o", col("amb_o").cast(FloatType())) \
    .withColumn("shar_o", col("shar_o").cast(FloatType()))

print("✓ Data types fixed")

# COMMAND ----------

# ── STEP 3: IMPUTE MISSING VALUES ──────────────────────────────────────────
# Imputation = filling in missing values intelligently
# We can't just delete rows with nulls — we'd lose too much data
# Strategy: fill numeric rating columns with the column mean
# Think of it as: "if someone didn't rate attractiveness,
# assume they'd give an average rating"
#
# .agg() = aggregate function (runs across the whole column)
# mean() = average of all non-null values
# .collect()[0][0] = pull the single result out of the Spark DataFrame
#   into a regular Python float we can use

rating_cols = ["attr", "sinc", "intel", "fun", "amb", "shar",
               "attr_o", "sinc_o", "intel_o", "fun_o", "amb_o", "shar_o"]

# Build a dictionary of {column_name: mean_value} for all rating columns
# We compute all means in one pass for efficiency
means = speed_df.agg(*[mean(c).alias(c) for c in rating_cols]).collect()[0]

# Fill each column's nulls with its own mean
# .fillna() takes a dictionary of {column: fill_value}
fill_values = {c: float(means[c]) for c in rating_cols if means[c] is not None}
speed_df = speed_df.fillna(fill_values)

print("✓ Missing values imputed with column means")

# COMMAND ----------

# ── STEP 4: FILTER OUT INVALID ROWS ───────────────────────────────────────
# Some rows have no match label at all — these are useless for ML
# If we don't know whether they matched, we can't train on that row
# .filter() keeps only rows where the condition is TRUE
# isnull() returns True if the value is null

speed_df = speed_df.filter(col("match").isNotNull())
print(f"After removing null match labels: {speed_df.count()} rows")

# COMMAND ----------

# ── STEP 5: SELECT COLUMNS WE ACTUALLY NEED ───────────────────────────────
# The speed dating dataset has 195 columns but we don't need all of them
# We select the ones relevant to our compatibility prediction
# Keeping only what we need makes the Silver table smaller and faster
# iid = person ID, pid = partner ID, match = our target label
# gender, age, age_o, race, race_o = demographics
# attr through shar_o = the six mutual rating attributes
# int_corr = correlation of interests score (pre-computed in dataset)
# samerace = whether the pair share the same racial background

speed_df = speed_df.select(
    "iid",          # person unique ID
    "pid",          # partner unique ID
    "match",        # TARGET LABEL — 1 if both said yes, 0 otherwise
    "gender",       # 0 = female, 1 = male
    "age",          # person's age
    "age_o",        # partner's age
    "race",         # person's race
    "race_o",       # partner's race
    "samerace",     # 1 if same race, 0 if not
    "int_corr",     # correlation between their stated interests
    "attr",         # person's rating of partner: attractiveness
    "sinc",         # person's rating of partner: sincerity
    "intel",        # person's rating of partner: intelligence
    "fun",          # person's rating of partner: fun
    "amb",          # person's rating of partner: ambition
    "shar",         # person's rating of partner: shared interests
    "attr_o",       # partner's rating of person: attractiveness
    "sinc_o",       # partner's rating of person: sincerity
    "intel_o",      # partner's rating of person: intelligence
    "fun_o",        # partner's rating of person: fun
    "amb_o",        # partner's rating of person: ambition
    "shar_o",       # partner's rating of person: shared interests
    "like",         # overall how much person liked partner
    "like_o",       # overall how much partner liked person
    "prob",         # person's estimate of partner saying yes
    "prob_o"        # partner's estimate of person saying yes
)

print(f"✓ Selected {len(speed_df.columns)} relevant columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Clean OKCupid Dataset
# MAGIC
# MAGIC OKCupid has 68,371 profiles and 2,620 columns.
# MAGIC Most columns are survey question answers — very sparse (lots of nulls).
# MAGIC We focus on the core profile features that overlap conceptually
# MAGIC with the speed dating attributes.

# COMMAND ----------

# ── STEP 1: SELECT CORE PROFILE COLUMNS ───────────────────────────────────
# With 2,620 columns we need to be selective
# We pick columns that capture the same dimensions as speed dating:
# demographics, personality, lifestyle, and preferences
# These will become the basis for our compatibility feature engineering

okcupid_df = okcupid_df.select(
    "age",          # age of the user
    "sex",          # gender
    "orientation",  # sexual orientation
    "status",       # relationship status
    "height",       # height in inches
    "body_type",    # self reported body type
    "diet",         # dietary preferences
    "drinks",       # drinking habits
    "drugs",        # drug use habits
    "education",    # education level
    "ethnicity",    # ethnic background
    "income",       # annual income
    "job",          # occupation
    "location",     # city, state
    "offspring",    # views on having kids
    "pets",         # pet preferences
    "religion",     # religious beliefs
    "sign",         # astrological sign
    "smokes",       # smoking habits
    "essay0",       # self summary essay
    "essay1",       # what I'm doing with my life
    "essay2",       # what I'm good at
    "essay3",       # favorite books/movies/shows
    "essay4",       # six things I can't live without
    "essay5",       # thinking about a lot lately
    "essay6",       # on a typical friday night
    "essay7",       # most private thing willing to admit
    "essay8",       # what I'm looking for
    "essay9"        # you should message me if
)

print(f"✓ Selected {len(okcupid_df.columns)} OKCupid columns")

# COMMAND ----------

# ── STEP 2: CLEAN STRING COLUMNS ──────────────────────────────────────────
# String columns often have inconsistent casing and extra whitespace
# "Male", "male", "MALE" should all be the same thing
# lower() converts to lowercase
# trim() removes leading and trailing spaces
# We apply this to all categorical string columns

string_cols = ["sex", "orientation", "status", "body_type", "diet",
               "drinks", "drugs", "education", "ethnicity", "job",
               "offspring", "pets", "religion", "smokes"]

for c in string_cols:
    okcupid_df = okcupid_df \
        .withColumn(c, lower(trim(col(c))))

print("✓ String columns normalized to lowercase")

# COMMAND ----------

# ── STEP 3: CLEAN AGE ─────────────────────────────────────────────────────
# Age should be a reasonable number — filter out anything unrealistic
# Anyone under 18 or over 100 is likely a data error
# .filter() keeps rows where both conditions are true
# .between() is shorthand for >= and <=

okcupid_df = okcupid_df.filter(
    col("age").between(18, 100)
)

print(f"After age filter: {okcupid_df.count()} rows")

# COMMAND ----------

# ── STEP 4: HANDLE INCOME ─────────────────────────────────────────────────
# Income has -1 as a sentinel value meaning "prefer not to say"
# We replace -1 with null so it's treated as missing, not a real value
# when().otherwise() is Spark's if/else — like Excel's IF()
# when(condition, value_if_true).otherwise(value_if_false)

okcupid_df = okcupid_df.withColumn(
    "income",
    when(col("income") == -1, None)  # -1 means unknown → replace with null
    .otherwise(col("income"))         # keep all other values as-is
)

print("✓ Income sentinel values (-1) replaced with null")

# COMMAND ----------

# ── STEP 5: DEDUPLICATE ────────────────────────────────────────────────────
# OKCupid profiles should be unique per person
# Drop any exact duplicate rows

before = okcupid_df.count()
okcupid_df = okcupid_df.dropDuplicates()
after = okcupid_df.count()
print(f"✓ Removed {before - after} duplicate OKCupid profiles")
print(f"  Remaining: {after} profiles")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write Silver Delta Tables
# MAGIC
# MAGIC Both datasets are now clean and trustworthy.
# MAGIC Write them to the Silver layer in ADLS Gen2.

# COMMAND ----------

# ── WRITE SPEED DATING SILVER ──────────────────────────────────────────────
speed_df.write \
    .format("delta") \
    .mode("overwrite") \
    .save(f"{base_path}/silver/speed_dating")

print("✓ Speed Dating Silver written successfully")

# COMMAND ----------

# ── WRITE OKCUPID SILVER ───────────────────────────────────────────────────
okcupid_df.write \
    .format("delta") \
    .mode("overwrite") \
    .save(f"{base_path}/silver/okcupid")

print("✓ OKCupid Silver written successfully")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify Silver Tables

# COMMAND ----------

# ── READ BACK AND VERIFY ───────────────────────────────────────────────────
speed_silver = spark.read.format("delta").load(f"{base_path}/silver/speed_dating")
okcupid_silver = spark.read.format("delta").load(f"{base_path}/silver/okcupid")

print(f"✓ Silver Speed Dating: {speed_silver.count()} rows, {len(speed_silver.columns)} columns")
print(f"✓ Silver OKCupid: {okcupid_silver.count()} rows, {len(okcupid_silver.columns)} columns")

# Show a sample of the speed dating silver table
# .show() prints the first N rows in a readable table format
speed_silver.show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Silver Cleaning Complete
# MAGIC
# MAGIC Both datasets are now clean, typed correctly, and deduplicated.
# MAGIC Next step: 03_gold_feature_engineering.py
# MAGIC We will engineer compatibility features and join the two datasets
# MAGIC into one unified ML-ready training table.