# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/als-recommender. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to deploy the ALS recommender. 

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC 
# MAGIC With our model trained, it's now time to examine how it will be deployed to support the delivery of product recommendations to users. The typical pattern of deployment is to train the model and then generate recommendations for each user which are then persisted for fast retrieval. </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/als_model_deployment.png' width=500>
# MAGIC 
# MAGIC In examining this pattern, it's important to remember the ALS *model* is fundamentally two lower-ranked matrices, one focused on users and the other focused on products/items, that are brought together to estimate *ratings*. These matrices don't make for the kind of high-speed predictions we'd want in an online scenario.  Furthermore, if any of the input data changes, both matrices must be recomputed in full.  This leads naturally to a pattern of training followed by immediate generation of the full output set on a periodic basis.

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import mlflow

import pyspark.sql.functions as fn

import pandas as pd

# COMMAND ----------

# MAGIC %md ##Step 1: Move Model to Production
# MAGIC 
# MAGIC In our last notebook, we persisted our model to mlflow.  Given the discussion above, it's reasonable to define a workflow where the model is trained and immediately used to generate recommendation output in a manner that doesn't require you to retrieve the model from the mlflow registry as an intermediary step.  But because we have separated this pattern into two discrete steps with mlflow in the middle, we will now demonstrate how to retrieve the model from the registry:

# COMMAND ----------

# DBTITLE 1,Identify Most Recent Model Version
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
latest_model_info = client.search_model_versions(f"name='{config['model name']}'")[0]
model_version = latest_model_info.version
model_status = latest_model_info.current_stage

# COMMAND ----------

# DBTITLE 1,Elevate to Production
if model_status.lower() != 'production':
  client.transition_model_version_stage(
    name=config['model name'],
    version=model_version,
    stage='production'
    ) 

# COMMAND ----------

# DBTITLE 1,Retrieve Latest Production Model
# load latest production model
retrieved_model = mlflow.spark.load_model(f"models:/{config['model name']}/production")

# COMMAND ----------

# MAGIC %md Elevating the model to *production* status prior to retrieval isn't absolutely necessary.  You could instead retrieve the model using a URI structured as *models:/\<model_name>/\<model_version>*.  That said, many organizations will want to implement a review workflow before a model is blessed for production use, so hopefully the above steps help illustrate how this might be done programmatically when needed.

# COMMAND ----------

# MAGIC %md ##Step 2: Generate Recommendations
# MAGIC 
# MAGIC The Spark model retrieved from mlflow has been persisted as an MLLib pipeline.  A pipeline consists of one or more stages where data is passed from stage to stage in order to arrive at an output.  This approach allows models retrieved from mlflow to have a standard API and allows data transformation logic to be coupled with model prediction capabilities.  The downside of this approach is that our particular model's API which features a number of very useful methods for the generation of recommendations are a little more difficult to access.
# MAGIC 
# MAGIC To see this pipeline structure, let's examine the *stages* associated with our retrieved model:

# COMMAND ----------

# DBTITLE 1,Examine Model Stages
print(
  retrieved_model.stages
  )

# COMMAND ----------

# MAGIC %md You can see our model consists of one stage which is the ALS model itself.  Because of the simplicity of the inputs into an ALS model, its typical that such a pipeline would consist of just the one model element (which is why we didn't bother constructing a pipeline ourselves in the prior notebook).  To access the underlying model, we just keep in mind that the pipeline is more or less a collection and the we then pluck the original ALS model from it as so:

# COMMAND ----------

# DBTITLE 1,Retrieve Original Model and Examine Recommendation Methods
original_model = retrieved_model.stages[0]

# COMMAND ----------

# MAGIC %md We now have a couple paths for the generation of recommendations, but before jumping into the mechanics of how we might do this, it's important to carefully think through how our recommendations will be used as this will guide us in terms of how we might implement the following.
# MAGIC 
# MAGIC One strategy for generating recommendations is just to grab an exhaustive list of products, sorted based on preference, for each user.  This is conceptually simple but it will take quite a bit of time and/or computational resources to complete.  If our goal is to generate email marketing, we might limit our recommendations to just those products that are relevant to the campaigns we wish to run, dramatically reducing the number of items we need to score for each user.  Similarly, if our goal is to populate a landing page for a product category, we might score just the products in that category in order to serve that goal.  If we are making more general recommendations, such as a *based on your previous purchases* style of recommendation, we might be open to all products regardless of category but limit the results to the top *number* of products based on how that data would be displayed.  The bottom line here is that we want to be thoughtful in how we approach the challenge of making recommendations as this will save us time and money down the road.
# MAGIC 
# MAGIC 
# MAGIC With that in mind, let's look at how we might generate recommendations for products in a particular department. 
# MAGIC 
# MAGIC The pipeline retrieved directly from mlflow exposes a *transform* methods which allows us to submit users and products to be scored.  We can generate recommendations from it doing something like this:

# COMMAND ----------

# DBTITLE 1,Get User-Product Ratings for Department
# limit recommendations to just this department
department = 'bakery'

# get list of users
ratings = spark.table('user_product_purchases')
users = ratings.select('user_id').distinct()

# get products within a department
products = (
  spark
    .table('departments')
    .filter(fn.expr(f"department='{department}'"))
    .join(
      spark.table('products'),
      on='department_id'
      )
    .select('product_id')
  )

# cross join all users and products in dataset
users_products = users.crossJoin(products)

# retrieve ratings for user-product combinations
recs = (
  retrieved_model
    .transform(users_products)
    .withColumn('department', fn.lit(department))
    .select('department', 'user_id', 'product_id', 'prediction')
    .repartition(sc.defaultParallelism*10, ['user_id', 'department'])
    .cache()
  )

display(recs)

# COMMAND ----------

# MAGIC %md While this seems like a lot of work compared to the approach we'll examine next, it does make assembling a clean list of users and ordered products pretty simple, though computationally expensive:

# COMMAND ----------

# DBTITLE 1,Assemble Product Ratings by User (windowed function)
display(
  recs
    .withColumn('recs', fn.expr('collect_list(product_id) over(partition by user_id order by prediction desc)'))
    .groupBy('user_id', 'department')
      .agg( fn.max('recs').alias('recs') ) # get longest assembled list of recommendations per user
    .select('department', 'user_id', 'recs')
  )

# COMMAND ----------

# MAGIC %md To lower the computational requirements of doing this, we could implement a pandas UDF:

# COMMAND ----------

# DBTITLE 1,Assemble Product Ratings by User (pandas UDF) 
# define function to sort products into recommendations for each user
def assemble_recs(keys, data: pd.DataFrame) -> pd.DataFrame:
  
  # get user id on which this data is grouped
  user_id = keys[0]
  department = keys[1]
  
  # sort data on rating
  sorted = data[['product_id','prediction']].sort_values('prediction', ascending=False)
  
  # assemble list of sorted products
  recs = sorted['product_id'].to_list()
  
  # return data frame of recommended products for this user
  return pd.DataFrame([[user_id, department, recs]], columns=['user_id', 'department', 'recs'])


display(
  recs
    .groupBy('user_id','department')
      .applyInPandas(assemble_recs, 'user_id int, department string, recs array<int>')
    .select('department', 'user_id', 'recs')
  )

# COMMAND ----------

# MAGIC %md Another way to generate recommendations is to call one of the recommendation methods available through the original model embedded in the retrieved pipeline.  These methods include the following:

# COMMAND ----------

# DBTITLE 1,Examine Recommendation Methods
print(
  [d for d in dir(original_model) if d.startswith('rec')]
  )

# COMMAND ----------

# MAGIC %md The *recommendForAllItems* and *recommendForItemSubset* methods return some specified number of users with the highest preference for either all items in the training dataset or just the subset of items supplied, respectively.  The *recommendForAllUsers* and *recommendForUserSubset* methods do the opposite, returning some specified number of highest preferred products for each user.  These methods can be very useful in specific scenarios such as identifying which customers to target with item-specific messaging or which limited number of products to position with users.
# MAGIC 
# MAGIC The predictions returned by these method calls are organized as an ordered array of struct values, each of which contains the item/product id and its rating.  If you wish to use these methods to generate recommendations, you may wish to use strip the rating value from the predictions to just hold on to the item/product ids.  A pandas UDF such as the one below may be helpful in accomplishing this:

# COMMAND ----------

# DBTITLE 1,Remove Ratings from Recommendations (pandas UDF)
@fn.pandas_udf('array<int>')
def remove_ratings(data: pd.Series) -> pd.Series:
  
  # define function to clean individual recommendation set
  def _clean_recs(recs):
    # just get the product id for each recommendation
    results = [rec['product_id'] for rec in recs]  
    return results
  
  # apply cleaning function to each row in the series
  return data.apply(_clean_recs)
  
  
#display(
#  original_model
#    .recommendForAllItems()
#    .select(
#      'user_id', 
#      remove_ratings('recommendations').alias('rec') 
#      )
#  )

# COMMAND ----------

# MAGIC %md ##Step 3: Publish Recommendations
# MAGIC 
# MAGIC With our recommendations generated, we now need to publish them for use in downstream systems. Popular systems for the publication of these data include:
# MAGIC </p>
# MAGIC 
# MAGIC * Redis
# MAGIC * Azure Cosmos DB
# MAGIC * Mongo DB
# MAGIC * Couchbase
# MAGIC * Elasticsearch
# MAGIC </p>
# MAGIC 
# MAGIC Each of these systems has their own connectors and requirements for publication.  You can use the [following resource](https://docs.databricks.com/external-data/index.html#what-data-services-does-databricks-integrate-with) as a starting point for exploring how to publish data to these systems.

# COMMAND ----------

# MAGIC %md As you consider the scheduling of your recommender pipelines, carefully consider the output you wish to produce.  Most organizations will accumulate a large number of users that become inactive over time.  Instead of generating recommendations for all users in the system, consider tailoring your inputs to focus on your *active* users so that you generate outputs for the users more likely to consume them. 

# COMMAND ----------

# MAGIC %md © 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
