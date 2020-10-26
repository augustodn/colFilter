import pandas as pd
import numpy as np
from math import sqrt
import helper

COMPANIES = range(100)
CANDIDATES = 5000
MAX_NUM_EMPLOYEES = 500
MAX_SCORE = 10
COMPANIES_SUBSET = 100
"""
# Uncomment block if you want to generate a ratings list
ratings = helper.make_ratings_list(COMPANIES, MAX_NUM_EMPLOYEES,
                                   CANDIDATES, MAX_SCORE)
ratings.to_csv('ratings.csv', index=False)
"""
ratings = pd.read_csv('ratings.csv')

# Let's rank some candidates as the recommender input
input_candidates = []
size = np.random.randint(MAX_NUM_EMPLOYEES)
candidates_to_rank = np.random.randint(CANDIDATES, size=size)
# Get a list of unique candidates
candidates_to_rank = list(set(candidates_to_rank))

for candidate in candidates_to_rank:
    score = np.random.randint(low=1, high=MAX_SCORE+1)
    input_candidates.append([candidate, score])

input_candidates = pd.DataFrame(input_candidates,
                                columns=['candidateId', 'score'])

# Filter the companies who have ranked the candidates
companies_subset = ratings[ratings['candidateId'].isin(
                        input_candidates['candidateId'].tolist())]
companies_subset = companies_subset.groupby(['companyId'])

# Sort by companies which shares the most candidates in common
companies_subset = sorted(companies_subset,
                          key=lambda x: len(x[1]),
                          reverse=True)

# Compare with the top 100 results
if len(companies_subset) > COMPANIES_SUBSET:
    companies_subset = companies_subset[0:COMPANIES_SUBSET]

# Call the correlation algorithm
correlated_companies = helper.make_pearson_correlation(companies_subset,
                                                       input_candidates)

correlated_companies = pd.DataFrame.from_dict(correlated_companies,
                                              orient='index')
correlated_companies.columns = ['similarityIndex']
correlated_companies['companyId'] = correlated_companies.index
correlated_companies.index = range(len(correlated_companies))

# Get the 50 most similar companies to the input candidates subset
top_companies = correlated_companies.sort_values(
                    by='similarityIndex', ascending=False)[0:50]

top_companies = top_companies.merge(ratings,
                            left_on='companyId',
                            right_on='companyId',
                            how='inner')
# Weight the scores by the similarity index (ranging from 0 to 1)
top_companies['weightedScore'] = top_companies['similarityIndex'] * \
                                 top_companies['score']
top_companies.dropna(inplace=True)

# Sum the similarity index and weighted score per each candidate
# the ones which appears the most will have a higher total value
temp_top_companies = top_companies.groupby('candidateId')\
                                  .sum()[['similarityIndex','weightedScore']]
temp_top_companies.columns = ['sum_similarityIndex','sum_weightedScore']

# Create the recommendation dataframe
recommendations = pd.DataFrame()
# Now we take the weighted average
recommendations['recomScore'] = temp_top_companies['sum_weightedScore'] / \
                                  temp_top_companies['sum_similarityIndex']
recommendations['candidateId'] = temp_top_companies.index
# Order by the highest score
recommendations = recommendations.sort_values(by='recomScore',
                                                  ascending=False)
print(recommendations.head())

dbg = """
# Check if there are recommended candidates in the input dataframe
index = recommendations.head().index[0]
candidate = recommendations.loc[index, 'candidateId']
print('\n\n')
print(input_candidates.loc[input_candidates.candidateId == candidate])
"""
