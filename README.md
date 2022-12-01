## ALS Recommender System Intro

Recommender systems are becoming increasing important as companies seek better ways to select products to present to end users. In this solution accelerator, we will explore a form of collaborative filter referred to as a matrix factorization.  

Matrix factorization works by assembling a set of ratings for various products made by a set of users.  The large user x products matrix is decomposed into smaller user and product submatrices associated with some developer-specified number of latent factors.  In many ways, a matrix factorization is a dimension reduction technique but one where missing values in the original matrix are allowed.

When examining ratings for a large number of user and product combinations, most users will engage with a very smaller percentage of products.  This causes us to have a user x products matrix that is highly sparse. When we decompose this matrix into the submatrices, the two can be combined to *recreate* the original matrix in a manner that provides ratings estimates for all products, including those a user has not yet engaged.  This ability to fill-in the missing ratings forms the basis for recommending new products to a user.

Matrix factorization recommenders are frequently used in scenarios where we wish to suggest new and repeat purchase items to a user.  *People like you also bought ...*, *Products we think you'll like ...*, and *Based on your purchase history ...* styled recommendations are frequently delivered through this type of recommender.

The challenge in developing a matrix factorization recommender is the large amount of computational horsepower required to calculate the submatrices.  Alternating Least Squares (ALS) is one approach that decomposes the process into a series of incremental steps that can be implemented in a distributed manner. In this solution accelerator, we will train and deploy an ALS-based matrix factorization recommender using the ALS capabilities in Apache Spark to demonstrate how this is done.

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |

## Instruction

To run this accelerator, clone this repo into a Databricks workspace. Attach the `RUNME` notebook to any cluster running a DBR 11.0 or later runtime, and execute the notebook via Run-All. A multi-step-job describing the accelerator pipeline will be created, and the link will be provided. Execute the multi-step-job to see how the pipeline runs. The job configuration is written in the RUNME notebook in json format. The cost associated with running the accelerator is the user's responsibility.
