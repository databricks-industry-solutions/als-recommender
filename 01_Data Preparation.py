# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/als-recommender. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to prepare the data for use in the ALS recommender. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook, we will make accessible purchase history data which will be used as the basis for the construction of a matrix factorization recommender.  The dataset we will use is the [Instacart dataset](https://www.kaggle.com/c/instacart-market-basket-analysis), downloadable from the Kaggle website. We will make the data available through a set of queryable tables and then derive implied ratings from the data before proceeding to the next notebook.

# COMMAND ----------

# MAGIC %md ## Step 1: Data Preparation
# MAGIC 
# MAGIC The data in the Instacart dataset should be [downloaded](https://www.kaggle.com/c/instacart-market-basket-analysis) and uploaded to cloud storage. The cloud storage location should then be [mounted](https://docs.databricks.com/data/databricks-file-system.html#mount-object-storage-to-dbfs) to the Databricks file system as shown here:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_filedownloads.png' width=240>
# MAGIC 
# MAGIC **NOTE** The name of the mount point, file locations and database used is configurable within the *00_Intro & Config* notebook.
# MAGIC 
# MAGIC The individual files that make up each entity in this dataset can then be presented as a queryable table as part of a database with a high-level schema as follows:</p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/instacart_schema2.png' width=300>
# MAGIC 
# MAGIC We have automated this data preparation step for you in the notebook below and used a `/tmp/instacart_als` storage path throughout this accelerator in place of the mount path. 

# COMMAND ----------

# MAGIC %run "./util/data-extract"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.sql.types import *
import pyspark.sql.functions as fn
from pyspark.sql import window as w

# COMMAND ----------

# DBTITLE 1,Define Table Creation Helper Functions
def read_data(file_path, schema):
  df = (
    spark
      .read
      .csv(
        file_path,
        header=True,
        schema=schema
        )
    )
  return df

def write_data(df, table_name):
   _ = (
       df
        .write
        .format('delta')
        .mode('overwrite')
        .option('overwriteSchema','true')
        .saveAsTable(table_name)
       )  

# COMMAND ----------

# DBTITLE 1,Load the Data To Tables
# orders data
# ---------------------------------------------------------
orders_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('user_id', IntegerType()),
  StructField('eval_set', StringType()),
  StructField('order_number', IntegerType()),
  StructField('order_dow', IntegerType()),
  StructField('order_hour_of_day', IntegerType()),
  StructField('days_since_prior_order', FloatType())
  ])

orders = read_data(config['orders_path'], orders_schema)
write_data( orders, '{0}.orders'.format(config['database']))
# ---------------------------------------------------------

# products
# ---------------------------------------------------------
products_schema = StructType([
  StructField('product_id', IntegerType()),
  StructField('product_name', StringType()),
  StructField('aisle_id', IntegerType()),
  StructField('department_id', IntegerType())
  ])

products = read_data( config['products_path'], products_schema)
write_data( products, '{0}.products'.format(config['database']))
# ---------------------------------------------------------

# order products
# ---------------------------------------------------------
order_products_schema = StructType([
  StructField('order_id', IntegerType()),
  StructField('product_id', IntegerType()),
  StructField('add_to_cart_order', IntegerType()),
  StructField('reordered', IntegerType())
  ])

order_products = read_data( config['order_products_path'], order_products_schema)
write_data( order_products, '{0}.order_products'.format(config['database']))
# ---------------------------------------------------------

# departments
# ---------------------------------------------------------
departments_schema = StructType([
  StructField('department_id', IntegerType()),
  StructField('department', StringType())  
  ])

departments = read_data( config['departments_path'], departments_schema)
write_data( departments, '{0}.departments'.format(config['database']))
# ---------------------------------------------------------

# aisles
# ---------------------------------------------------------
aisles_schema = StructType([
  StructField('aisle_id', IntegerType()),
  StructField('aisle', StringType())  
  ])

aisles = read_data( config['aisles_path'], aisles_schema)
write_data( aisles, '{0}.aisles'.format(config['database']))
# ---------------------------------------------------------

# COMMAND ----------

# DBTITLE 1,Present Tables in Database
display(
  spark
    .sql('SHOW TABLES')
  )

# COMMAND ----------

# MAGIC %md ## Step 2: Generate Ratings
# MAGIC 
# MAGIC The records that make up the Instacart dataset represent grocery purchases. As would be expected in a grocery scenario, there are no explicit ratings provided in this dataset. Explicit ratings are typically found in scenarios where users are significantly invested (either monetarily or in terms of time or social standing) in the items they are purchasing or consuming.  When we are considering apples and bananas purchased to have around the house as a snack or to be dropped in a kid's lunch, most users are just not interested in providing 1 to 5 star ratings on those items.
# MAGIC 
# MAGIC We therefore need to examine the data for implied ratings (preferences).  In a grocery scenario where items are purchased for consumption, repeat purchases may provide a strong signal of preference. [Douglas Oard and Jinmook Kim](https://terpconnect.umd.edu/~oard/pdf/aaai98.pdf) provide a nice discussion of the various ways we might derive implicit ratings in a variety of scenarios and it is certainly worth considering alternative ways of deriving an input metric.  However, for the sake of simplicity, we'll leverage the percentage of purchases involving a particular item as our implied rating:

# COMMAND ----------

# MAGIC %sql
# MAGIC DROP VIEW IF EXISTS user_product_purchases;
# MAGIC 
# MAGIC CREATE VIEW user_product_purchases
# MAGIC AS
# MAGIC   SELECT
# MAGIC     monotonically_increasing_id() as row_id,
# MAGIC     x.user_id,
# MAGIC     x.product_id,
# MAGIC     x.product_purchases / y.purchase_events as rating
# MAGIC   FROM (  -- product purchases
# MAGIC     SELECT
# MAGIC       a.user_id,
# MAGIC       b.product_id,
# MAGIC       COUNT(*) as product_purchases
# MAGIC     FROM orders a
# MAGIC     INNER JOIN order_products b
# MAGIC       ON a.order_id=b.order_id
# MAGIC     INNER JOIN products c
# MAGIC       ON b.product_id=c.product_id
# MAGIC     GROUP BY a.user_id, b.product_id
# MAGIC     ) x 
# MAGIC   INNER JOIN ( -- purchase events
# MAGIC     SELECT 
# MAGIC       user_id, 
# MAGIC       COUNT(DISTINCT order_id) as purchase_events 
# MAGIC     FROM orders 
# MAGIC     GROUP BY user_id
# MAGIC     ) y
# MAGIC     ON x.user_id=y.user_id
# MAGIC     ;
# MAGIC     
# MAGIC SELECT *
# MAGIC FROM user_product_purchases;

# COMMAND ----------

# MAGIC %md © 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
