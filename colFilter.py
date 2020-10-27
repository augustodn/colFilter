import pandas as pd
import numpy as np
from math import sqrt
import helper

COMPANIES = range(100)
CANDIDATES = 5000
MAX_NUM_EMPLOYEES = 500
MAX_SCORE = 10
COMPANIES_SUBSET = 100
TOP_SIMILAR_COMP = 50

"""
# Uncomment block if you want to generate a ratings list
ratings = helper.make_ratings_list(COMPANIES, MAX_NUM_EMPLOYEES,
                                   CANDIDATES, MAX_SCORE)
ratings.to_csv('ratings.csv', index=False)
"""
ratings = pd.read_csv('ratings.csv')

# Let's rank some candidates as the recommender input
# These candidates are the current company employees
"""
# Uncomment block if you want to generate a ratings list
input_candidates = helper.make_input_list(
    MAX_NUM_EMPLOYEES, CANDIDATES, MAX_SCORE)
input_candidates.to_csv('input_candidates.csv', index=False)
"""
input_candidates = pd.read_csv('input_candidates.csv')

# Filter the companies who have ranked the candidates
companies_subset = ratings[ratings['candidateId'].isin(
                        input_candidates['candidateId'].tolist())]
companies_subset = companies_subset.groupby(['companyId'])

# Sort by companies which share the most candidates in common
companies_subset = sorted(companies_subset,
                          key=lambda x: len(x[1]),
                          reverse=True)

# Compare with the top results
if len(companies_subset) > COMPANIES_SUBSET:
    companies_subset = companies_subset[0:COMPANIES_SUBSET]

# Call the correlation algorithm
correlated_companies = helper.make_pearson_correlation(companies_subset,
                                                       input_candidates)

correlated_companies = pd.DataFrame.from_dict(correlated_companies,
                                              orient='index')
correlated_companies.columns = ['similarityIndex']
correlated_companies['companyId'] = correlated_companies.index

# Remove companies with negative or no correlation
no_correlation = correlated_companies.loc[
                        correlated_companies.similarityIndex <= 0]
correlated_companies = correlated_companies.drop(no_correlation.index)

# Get the most similar companies and sort them
top_companies = correlated_companies.sort_values(
                    by='similarityIndex', ascending=False)[0:TOP_SIMILAR_COMP]

top_companies = top_companies.merge(ratings,
                            left_on='companyId',
                            right_on='companyId',
                            how='inner')

# Weight the scores by the similarity index (ranging from 0 to 1)
top_companies['weightedScore'] = top_companies['similarityIndex'] * \
                                 top_companies['score']
top_companies.dropna(inplace=True)

# Sum the similarity index and weighted score per each candidate ocurrence
# the ones which appears the most will have a higher total value
temp_top_companies = top_companies.groupby('candidateId')\
                                  .sum()[['similarityIndex','weightedScore']]
temp_top_companies.columns = ['sum_similarityIndex','sum_weightedScore']

recommendations = pd.DataFrame()
# Now we take the weighted average
# Changed matematical operation: * instead of /
recommendations['recomScore'] = temp_top_companies['sum_weightedScore'] * \
                                temp_top_companies['sum_similarityIndex']
recommendations['candidateId'] = temp_top_companies.index
# Order by the highest score
recommendations = recommendations.sort_values(by='recomScore',
                                                  ascending=False)

# Clean the actual input_candidates from the recommendations
actual_employees = recommendations[recommendations['candidateId']\
                   .isin(input_candidates['candidateId'].tolist())]
recommendations = recommendations.drop(actual_employees.index)

print(recommendations.head(10))
