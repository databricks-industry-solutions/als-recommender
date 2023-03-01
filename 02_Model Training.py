# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/als-recommender. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/recommendation-engines

# COMMAND ----------

# MAGIC %md The purpose of this notebook is to train the ALS recommender. 

# COMMAND ----------

# MAGIC %md ## Introduction
# MAGIC 
# MAGIC In this notebook, we'll train a matrix factorization recommender using the  Alternating Least Squares (ALS) algorithm built into Spark. We'll start by working through the mechanics of model training and evaluation, pivot into a hyperparameter tuning exercise and then train a final model using optimized parameter settings. 

# COMMAND ----------

# DBTITLE 1,Get Config Info
# MAGIC %run "./00_Intro & Config"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.recommendation import ALS

import mlflow

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
import numpy as np

import pyspark.sql.window as w
import pyspark.sql.functions as fn
from pyspark.sql.types import *

import random

# COMMAND ----------

# MAGIC %md ## Step 1: Retrieve Data
# MAGIC 
# MAGIC In the last notebook, we calculated implied ratings for those items purchased by users.  We will retrieve those ratings and then take a random sample on which will perform our tuning work:

# COMMAND ----------

# DBTITLE 1,Retrieve Data
# retrieve all ratings
ratings = spark.table('user_product_purchases')

# retrieve sampling of ratings
ratings_sampled = ratings.sample(0.10).cache()

print(f'Total:\t{ratings.count()}')
print(f'Sample:\t{ratings_sampled.count()}')

# COMMAND ----------

# MAGIC %md In writing the cell above, we had quite a bit of debate about the sampling approach.  In our tests, we found that the 10% random sample did not impact the sparsity of the user-product matrix on which we intended to tune our model.  However, with smaller sample sizes we did find that matrix sparsity, *i.e.* the number of user-product combinations with values relative to the total number of possible user-product combinations in the resulting matrix, was altered in a way that made us concerned about the representativeness of the sample.  
# MAGIC 
# MAGIC In order to preserve products, we had discussed taking a random sample of users and bringing ratings forward for all products in constructing the sampled matrix.  This is a common approach when developing such samples for tuning purposes.  However, with this dataset, it was easier to just bump up the sample size to avoid the concern.
# MAGIC 
# MAGIC As you consider sampling for model tuning, please examine the sparsity of your original matrix and the sampled matrix to ensure you are not altering the ratio of rated to potential combinations in a meaningful way.
# MAGIC 
# MAGIC With our datasets in hand, we'll split each into training and testing datasets as is standard practice in model training exercises:

# COMMAND ----------

# DBTITLE 1,Generate Train-Test Split
ratings_train, ratings_test = ratings.randomSplit([0.8, 0.2], random.randrange(100000))
ratings_sampled_train, ratings_sampled_test = ratings.randomSplit([0.8, 0.2], random.randrange(100000))

# COMMAND ----------

# MAGIC %md ## Step 2: Define the Model
# MAGIC 
# MAGIC The [Alternating Least Squares model](https://spark.apache.org/docs/latest/api/python/reference/api/pyspark.ml.recommendation.ALS.html) implemented in PySpark MLLib is fairly straightforward.  Given a set of ratings associated with users and items, it will construct a matrix.  It will attempt to decompose this matrix into two lower ranked (factor) matrices each of which is centered on either the user or the items in the original ratings matrix.  When multiplied together, the resulting matrix not only provides good approximations of the original ratings but includes estimates for ratings values where there were gaps in the original.
# MAGIC 
# MAGIC How well the user and product (factor) matrices perform is a factor of the original data as well as the size of the matrices.  Each of the lower level matrices is sized according to the number of users or items and the number of latent factors or the *rank* we wish to assign them. In general, the higher the rank, the better the results but the higher the complexity of the model and therefore the longer it takes to calculate.
# MAGIC 
# MAGIC The decomposition of our ratings matrix into factor matrices is referred to as matrix factorization.  The process by which the factor matrices are estimated is referred to as alternating least squares (ALS). In the ALS algorithm, a user matrix is defined as a constant while the product matrix is optimized to minimize the (squared) error that occurs as the user and product matrices are combined to estimate the original ratings matrix.  Once this is done, the product matrix is then held constant while the user matrix is optimized.  This process goes back and forth until a stable solution is found or the number of allowed iterations has been completed. A regularization parameter is used to avoid overfitting in this process. 
# MAGIC 
# MAGIC The optimization depends on a comparison of estimated ratings against actual ratings. Because not every user has a rating for every product in the original rating matrix, only a subset of user and products need to be explored during the ALS process.  In the MLLib implementation of ALS, users and products are divided into blocks that minimize the amount of user or product information that needs to move between the blocks as part of the back-and-forth of the ALS process.  
# MAGIC 
# MAGIC **NOTE** As will be discussed below, our objective isn't exactly to recreate the original ratings but instead to generate scores that will position items in an appropriate order in accordance to perceived user preferences. With that in mind, you may see that your predicted ratings deviate some from original ratings and predictions may even dip into negative values.  This is to be expected.
# MAGIC 
# MAGIC With that in mind, let's train an initial ALS model.  The *userCol* and *itemCol* parameters identify the fields in our dataframe that represent the user and item identities, respectively.  The *ratingCol* identifies the field containing the rating provided by a user for a given item.  And because we are training the model to predict implicit preferences, we set [*implicitPrefs*](https://spark.apache.org/docs/latest/ml-collaborative-filtering.html#explicit-vs-implicit-feedback) to *True*:

# COMMAND ----------

# DBTITLE 1,Train an Initial Model
# define the als model
als = ALS(
  rank=100,
  maxIter=5,  
  userCol='user_id', 
  itemCol='product_id', 
  ratingCol='rating',
  implicitPrefs=True
  )

# train the model
model = als.fit(ratings_sampled_train)

# COMMAND ----------

# MAGIC %md Using our model, we can get predictions for our testing set as follows:

# COMMAND ----------

# DBTITLE 1,Make Predictions
predictions = model.transform(ratings_sampled_test)

display(predictions)

# COMMAND ----------

# MAGIC %md You may notice in the predictions above that there are several user-product combinations that return a prediction of *NaN*.  Our model has been trained on a subset of users and products and not every user or product in the test set made it into the training set.  Because of this, we don't have latent factors for those items which makes it impossible for us to generate predictions for user-product combinations involving them.  By default, the model returns these predictions as *NaN* values.
# MAGIC 
# MAGIC You can change this behavior by setting the model's **coldStartStrategy** setting to *drop*.  This will cause those values to be dropped from the returned set of predictions:

# COMMAND ----------

# DBTITLE 1,Make Predictions (Dropping Unknown Users & Products)
model.setColdStartStrategy('drop')

predictions = model.transform(ratings_sampled_test)

display(predictions)

# COMMAND ----------

# MAGIC %md So, how should we evaluate our model? Remember that we are working with implicit ratings.  In the [original paper](https://ieeexplore.ieee.org/document/4781121) on which the ALS algorithm is based, the authors spend quite a bit of time explaining how implicit ratings aren't actual ratings but instead are indicators of a binary preference, *i.e.*  the user prefers this product (1) or does not (0).  Our implicit ratings are better understood as measures of confidence that a user has a preference for a product.  That confidence can be used to suggest how confident we might be in our own recommendation of a product to a user where we would place the items we are most confident in recommending higher up in the list of our recommendations.
# MAGIC 
# MAGIC This might feel like we are playing a bit with language, but this concept affects how we go about calculating the values that form our predictions. When we set the *implicitPrefs* argument to *True*, we told the model to be less concerned with getting predicted ratings close to the provided ratings value and instead to focus on getting predicted ratings in an appropriate order relative to each other.
# MAGIC 
# MAGIC Because of this, we can't really evaluate this model based on predicted values relative to provided ratings (as would typically be done with a mean squared error or similar metric) but instead in terms of the sequence of items recommended in terms of the sequence of items actually selected by the user.  In other words, we need a ranking evaluator.
# MAGIC 
# MAGIC The most commonly employed ranking evaluator is *MAP@K*.  MAP@K stands for *mean average precision @ k* and it takes a little work to develop an intuition around the metric.  If you want to skip that part, just know that MAP@K improves as we move from 0.0 closer to 1.0.
# MAGIC 
# MAGIC Okay, so how do we interpret MAP@K?  Consider a list of *k* number of items we might recommend to a user.  These items are sequenced from most likely to be preferred to least likely.  The *k* number of items reflects the number of items we actually intend to present the user though most evaluators set this value to something like 5 to 10 to focus on how well we get the preferred items at the very top of any list of recommendations.
# MAGIC 
# MAGIC Looking over this list of *k* items, we start with the first position and ask if this item was a selected item.  If it is, it gets a *precision* of 1 at k=1.  If it's not, it gets a precision of 0.  We then look at the first and second positions and ask how many of the presented items were selected.  If all were selected, we again get a *precision* score of 1 at k=2.  If none were selected, we get a precision score of 0 at k=2.  And if only one was selected, we get a precision score of 1/2 or 0.5, again at k=2. We continue looking at the first through third, then the first through fourth and so on and so on until we've calculated *precision* scores for each first through *k*th position. 
# MAGIC 
# MAGIC We then average those precision scores to get the *average precision* for the recommendations for this particular user. If we repeat this calculation of *average precision* for each user, can then average the average precision scores across all users to arrive at a *mean average precision* score across the dataset. That's our *mean average precision @ k* metric.
# MAGIC 
# MAGIC The challenge with MAP@K as an evaluation metric is that it sets an incredibly high bar for selections.  It also is focused on items we've actually selected in the past and in some ways is penalizing us for suggesting new products.  The trick in working with MAP@K is to accept that you're likely to produce lower scores for most recommenders.  Our goal isn't necessarily to push MAP@K to 1.0, but instead to use the metric to compare different recommenders for their relative performance.  In other words, don't evaluate a recommender as good or bad based on its MAP@K score.  Consider its value in driving a higher MAP@K score relative to your next best recommender option.
# MAGIC 
# MAGIC To calculate MAP@K for our recommender, we need to decide a value for *k*.  We might choose 10 as that seems like a reasonably sized list of items to present and below that position (depending on our application) we might expect the user to enter into more of a browsing mode of item engagement that depends less on recommendation strength.  We can then ask our model to recommend the top 10 items for each user as follows:

# COMMAND ----------

# DBTITLE 1,Get Top 10 Recommendations per User
display(
  model.recommendForAllUsers(10)
  )


# COMMAND ----------

# MAGIC %md It's a little frustrating that the ALS model doesn't return the recommendations in the format required by our ranking evaluator.  So, we'll need to strip out just the *product_id* values from the array of recommendations while preserving the sequence of those recommendations.  We can do that by first exploding our recommendations in a manner that generates a column capturing the position of the resulting value in our original array.  From there, we will rebuild our list of products in the right sequence using a windowed version of the *collect_list* function.  The *order by* clause in that window definition will cause a list of one value to be generated for our first item, and list of two values for our second item, and so on and so on.  For that reason, we'll get the largest of our lists for each using a *max* aggregation:

# COMMAND ----------

# DBTITLE 1,Get Top 10 Recommendations per User (Just Product IDs)
display(
  model
    .recommendForAllUsers(10)
    .select( 
      'user_id',
      fn.posexplode('recommendations').alias('pos', 'rec') 
      )
    .withColumn('recs', fn.expr("collect_list(rec.product_id) over(partition by user_id order by pos)"))
    .groupBy('user_id')
      .agg( fn.max('recs').alias('recs')) 
  )

# COMMAND ----------

# MAGIC %md Now we get our actuals:

# COMMAND ----------

# DBTITLE 1,Get Actual Selections
display(
  ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(product_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr("size(selections)<=10"))
    .groupBy('user_id')
      .agg(
        fn.max('selections')
        )
    )

# COMMAND ----------

# MAGIC %md Now we can combine our actuals and predicted selections to perform the MAP@K evaluation:
# MAGIC 
# MAGIC **NOTE** Even though our item column values are integers and will typically be so, the ranking evaluator expects these values to be delivered as double-precision floating point values.  We've added a cast statement to each dataset definition to tackle this.

# COMMAND ----------

# DBTITLE 1,Calculate Map @ 10
k = 10

predicted = (
  model
    .recommendForAllUsers(k)
    .select( 
      'user_id',
      fn.posexplode('recommendations').alias('pos', 'rec') 
      )
    .withColumn('recs', fn.expr("collect_list(rec.product_id) over(partition by user_id order by pos)"))
    .groupBy('user_id')
      .agg( fn.max('recs').alias('recs'))
    .withColumn('prediction', fn.col('recs').cast('array<double>')) # cast the data to the types expected by the evaluator
  )

actuals = (
  ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(product_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>')) # cast the data to the types expected by the evaluator
  )

# evaluate the predictions
eval = RankingEvaluator( 
  predictionCol='prediction',
  labelCol='label',
  metricName='precisionAtK',
  k=k
  )

eval.evaluate( predicted.join(actuals, on='user_id') )

# COMMAND ----------

# MAGIC %md The MAP@K value above is quite low when we consider perfect MAP@K is 1.0.  That said, our goal is not to push towards a perfect score but instead to use MAP@K to compare the performance of models relative to each other.   

# COMMAND ----------

# MAGIC %md ##Step 3: Tune Hyperparamters
# MAGIC 
# MAGIC We now have all the elements in place to start tuning our model.  With that in mind, let's look at some of the model parameters we previously ignored.  The critical ones in terms of prediction quality are as follows:
# MAGIC </p>
# MAGIC 
# MAGIC * **maxIter** - the number of cycles between user and item optimizations to employ in training the model. The more cycles we give, the better the predictions but the longer the training time.
# MAGIC * **rank** - the number of latent factors to calculate for each of the user and item submatrices
# MAGIC * **regParam** - the regularization parameter controlling the gradient decent algorithm used during latent factor optimization. Should this be  greater than 0.0 and as high as 1.0.  There's an interesting discussion on this parameter in [this white paper](https://doi.org/10.1007/978-3-540-68880-8_32).
# MAGIC * **alpha** - the parameter multiplied against our implicit ratings in order to expand the influence of high scores.  This factor is often 1 or (much) higher.
# MAGIC * **nonnegative** - allow predicted values to go negative.  If you are using the predictions as simply a ranking mechanism (as we are doing here), leave this at its default value of False.
# MAGIC </p>
# MAGIC 
# MAGIC **NOTE** The higher the **rank** parameter (as well as the **maxIter** parameter up to a point), the better your model should perform (up to a point) but the longer it will take to process.  Instead of performing hyperparameter tuning on *rank* (which should almost always gravitate to the highest value you will allow within a given set of time constraints), consider testing processing time duration and model performance and make a decision about the highest value you want to fix for that parameter. (Same goes for *maxIter*.)
# MAGIC 
# MAGIC Other parameters associated with the model affect training performance.  These include:
# MAGIC </p>
# MAGIC 
# MAGIC * **blockSize** - This controls the preferred block size of the model.  More details on this can be reviewed [here](https://issues.apache.org/jira/browse/SPARK-20443), but you'll typically leave this value alone.
# MAGIC * **numItemBlocks** and **numUserBlocks** - the number of blocks to employ as part of the distributed computation of either the item or user matrices.  The default value is 10. You might play with these values to see how they affect performance with matrices of different sizes and complexity but we'll leave these alone.
# MAGIC </p>
# MAGIC 
# MAGIC 
# MAGIC With these parameters in mind, let's define a hyperparameter search space that we can use with an intelligent search conducted by [hyperopt](https://docs.databricks.com/machine-learning/automl-hyperparam-tuning/index.html).  To learn more about the definition of hyperopt search spaces, please refer to [this document](http://hyperopt.github.io/hyperopt/getting-started/search_spaces/).  We'd also suggest you take a look at [this excellent blog post](https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html) on how to get the most out of hyperopt:

# COMMAND ----------

# DBTITLE 1,Define Hyperparameter Search Space
search_space = {
  'regParam': hp.uniform('regParam', 0.01, 0.5),
  'alpha':hp.uniform('alpha', 1.0, 10.0)
  }

# COMMAND ----------

# MAGIC %md Now we can write a function to train and evaluate our model against various hyperparameter values retrieved from our search space:
# MAGIC 
# MAGIC **NOTE** We found that setting the *numItemBlocks* and *numUserBlocks* to a value aligned with the number of executors in our cluster helped speed up performance.  We have not performed an exhaustive test of this approach and expect that results will vary with cluster size and specific datasets. 

# COMMAND ----------

# DBTITLE 1,Define Evaluation Function for Tuning Trials
# define model to evaluate hyperparameter values
def evaluate(params):
  
  # clean up params
  if 'maxIter' in params: params['maxIter']=int(params['maxIter'])
  if 'rank' in params: params['rank']=int(params['rank'])
  
  with mlflow.start_run(nested=True):
    
    # instantiate model
    als = ALS(
      rank=100,
      maxIter=20,
      userCol='user_id',  
      itemCol='product_id', 
      ratingCol='rating', 
      implicitPrefs=True,
      numItemBlocks=sc.defaultParallelism,
      numUserBlocks=sc.defaultParallelism,
      **params
      )
    
    # train model
    model = als.fit(ratings_sampled_train)
    
    # generate recommendations
    predicted = (
      model
        .recommendForAllUsers(k)
        .select( 
          'user_id',
          fn.posexplode('recommendations').alias('pos', 'rec') 
          )
        .withColumn('recs', fn.expr("collect_list(rec.product_id) over(partition by user_id order by pos)"))
        .groupBy('user_id')
          .agg( fn.max('recs').alias('recs'))
        .withColumn('prediction', fn.col('recs').cast('array<double>'))
      )
    
    # score the model 
    eval = RankingEvaluator( 
      predictionCol='prediction',
      labelCol='label',
      metricName='precisionAtK',
      k=k
      )
    mapk = eval.evaluate( predicted.join(actuals, on='user_id') )
    
    # log parameters & metrics
    mlflow.log_params(params)
    mlflow.log_metrics( {'map@k':mapk} )
    
  return {'loss': -1 * mapk, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md There's a lot going on in our function, so let's break it down a bit.  First, we receive our hyperparameter values from the search space.  Because some of the values are selected from a range, they come in as floats when they are expected to be integers, so we've got a little clean up to do.
# MAGIC 
# MAGIC We then train our model using these hyperparameter values.  The trained model is used to generate predictions which are then evaluated to get a MAK@K score.  We log the evaluation metric and the hyperparameter values that lead to that to mlflow so we can examine these later.  And we then return our evaluation metric back to hyperopt for this run.  Notice that hyperopt is expecting a loss value to minimize.  Because MAP@K is better as it increases, we just flip it to a negative value to get hyperopt to work with it properly.
# MAGIC 
# MAGIC Notice too that we are referencing some variables such as *k* and *actuals* that aren't defined in the function.  We'll define these now to make them accessible to our function: 

# COMMAND ----------

# DBTITLE 1,Define Variables Used Between Runs
# k for map at k evaluations
k = 10

# calculate actuals once and cache for faster evaluations
actuals = (
  ratings_sampled_test
    .withColumn('selections', fn.expr("collect_list(product_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>'))
  ).cache()

# COMMAND ----------

# MAGIC %md Now we can bring everything together to perform our training runs.  Here we ask hyperopt to use our training function to evaluate values from our search space.  With each of the evaluation cycles, hyperopt considers the results and adjusts its search to hone in on an optimal set of hyperparameter values.  
# MAGIC 
# MAGIC Notice that we are using *Trails()* and not *SparkTrails()* with our hyperopt run.  *SparkTrails()* will attempt to parallelize our hyperparameter tuning runs across a Databricks cluster, but we are already making use of the Spark MLLib ALS model which is itself distributed.  You can only employ one parallelization pattern at a time so we leverage distributed model training and hyperopt run once cycle at a time across our cluster:

# COMMAND ----------

# DBTITLE 1,Tune the Model
# disable model logging at this phase
mlflow.autolog(exclusive=False, log_models=False) # https://docs.databricks.com/mlflow/databricks-autologging.html

# perform training runs
with mlflow.start_run(run_name='als_hyperopt_run'):
  
  argmin = fmin(
    fn=evaluate,
    space=search_space,
    algo=tpe.suggest,
    max_evals=20,
    trials=Trials()
    )
  
# report on best parameters discovered
print(space_eval(search_space, argmin))

# COMMAND ----------

# MAGIC %md As mentioned earlier, we are logging each trail to mlflow.  In Databricks, logging is enabled by default with hyperopt though we can configure it to not record the model itself as we did in the code above.  
# MAGIC 
# MAGIC Clicking on the mlflow tracking (flask) icon towards the top-right of the notebook and then selecting the view experiments icon in the upper right of the resulting pane, we can see details about the different model runs and perform comparisons to understand how different parameters affect our evaluation metrics.  More details on this can be found [here](https://docs.databricks.com/mlflow/tracking.html), but this data can be used to help us narrow our search space so that future iterations can spend more time scrutinizing the most productive regions of the space:
# MAGIC </p>
# MAGIC 
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/als_mlflow_alpha2.PNG'>

# COMMAND ----------

# MAGIC %md ##Step 4: Train and Persist Final Model
# MAGIC 
# MAGIC With our hyperparameters selected, we can now proceed to perform a final training run against our full set of data and then persist our model for later use. Notice that we are using the full set of ratings data and evaluating over the same.  Its important that our model have full access to implicit ratings available to us as we are limited in our ability to generate ratings for users and products on which the model is trained. In this regard, our model is really more of a transformation than a true predictive model capable of making generalizations:

# COMMAND ----------

# DBTITLE 1,Train Final Model
k = 10

actuals = (
  ratings
    .withColumn('selections', fn.expr("collect_list(product_id) over(partition by user_id order by rating desc)"))
    .filter(fn.expr(f"size(selections)<={k}"))
    .groupBy('user_id')
      .agg(
        fn.max('selections').alias('selections')
        )
    .withColumn('label', fn.col('selections').cast('array<double>'))
  ).cache()

# get parameters from prior tuning run
params = space_eval(search_space, argmin)
if 'maxIter' in params: params['maxIter']=int(params['maxIter'])
if 'rank' in params: params['rank']=int(params['rank'])

with mlflow.start_run(run_name='als_full_model'):

  # instantiate model
  als = ALS(
    rank=100,
    maxIter=50,
    userCol='user_id',  
    itemCol='product_id', 
    ratingCol='rating', 
    implicitPrefs=True,
    numItemBlocks=sc.defaultParallelism,
    numUserBlocks=sc.defaultParallelism,
    **params
    )

  # train model
  model = als.fit(ratings)

  # generate recommendations
  predicted = (
    model
      .recommendForAllUsers(k)
      .select( 
        'user_id',
        fn.posexplode('recommendations').alias('pos', 'rec') 
        )
      .withColumn('recs', fn.expr("collect_list(rec.product_id) over(partition by user_id order by pos)"))
      .groupBy('user_id')
        .agg( fn.max('recs').alias('recs'))
      .withColumn('prediction', fn.col('recs').cast('array<double>'))
    )

  # perform evaluation
  eval = RankingEvaluator( 
    predictionCol='prediction',
    labelCol='label',
    metricName='precisionAtK',
    k=k
    )
  
  mapk = eval.evaluate( predicted.join(actuals, on='user_id') )

  # log model details
  mlflow.log_params(params)
  mlflow.log_metrics( {'map@k':mapk} )
  mlflow.spark.log_model(model, artifact_path='model', registered_model_name=config['model name'])

# COMMAND ----------

# MAGIC %md Our model is now registered with mlflow using the name *als*.  With each run of the cell above, a new version of the model is registered.  The version number can be used to allow us to track the model as it moves through subsequent consideration for a production deployment.  We'll retrieve that version number here before then moving on to deployment steps in the next notebook:

# COMMAND ----------

# DBTITLE 1,Get Persisted Model Version Number
# connect to mlflow
client = mlflow.tracking.MlflowClient()

# identify model version in registry
model_version = client.search_model_versions(f"name='{config['model name']}'")[0].version

model_version

# COMMAND ----------

# MAGIC %md © 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License.
