import pandas as pd
import numpy as np
from scipy import stats

def make_pearson_correlation(companies_subset, input_candidates):
    pearsonCorrelationDict = {}

    # For every company group in our subset
    for company, group in companies_subset:
        # Let's start by sorting the input and current company group so the values aren't 
        # mixed up later on
        group = group.sort_values(by='candidateId')
        input_candidates = input_candidates.sort_values(by='candidateId')

        # Get the N for the formula
        nRatings = len(group)
        # Get the review scores for the candidates that both copmanies have in common
        temp_df = input_candidates[input_candidates['candidateId'].\
                    isin(group['candidateId'].tolist())]
        # And then store them in a temporary buffer variable in a list format to 
        # facilitate future calculations
        tempRatingList = temp_df['score'].tolist()
        # Let's also put the current user group reviews in a list format
        tempGroupList = group['score'].tolist()

        # Let the lists have the same length
        group_len = len(tempGroupList)
        rating_len = len(tempRatingList)

        if group_len > rating_len:
            tempGroupList = tempGroupList[:rating_len]
        if rating_len > group_len:
            tempRatingList = tempRatingList[:group_len]
        # Make the correlation
        if len(tempGroupList) > 10:
            correlation_factor, _ = stats.pearsonr(tempRatingList, tempGroupList)
        else:
            # If the lists have only one element correlation cannot be
            # determined
            correlation_factor= 0

        pearsonCorrelationDict[company] = correlation_factor

    return pearsonCorrelationDict

def make_ratings_list(companies, max_num_employees, candidates, max_score):

    ratings = []

    for company in companies:
        size = np.random.randint(max_num_employees)
        candidates_to_rank = np.random.randint(candidates, size=size)
        # Get a list of unique candidates
        candidates_to_rank = list(set(candidates_to_rank))

        for candidate in candidates_to_rank:
            score = np.random.randint(low=1, high=max_score+1)
            ratings.append([company, candidate, score])

    ratings = pd.DataFrame(ratings, columns=['companyId', 'candidateId', 'score'])

    return ratings
