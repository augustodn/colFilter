import pandas as pd
import numpy as np
from scipy import stats

def make_pearson_correlation(companies_subset, input_candidates):
    pearson_correlation = {}

    # For every company group in our subset
    for company, group in companies_subset:
        # Let's start by sorting the input and current company group so the values 
        # aren't mixed up later on
        group = group.sort_values(by='candidateId')
        input_candidates = input_candidates.sort_values(by='candidateId')

        # Get the review scores for candidates that appears in both companies
        # DEBUG: 
        # ir = input_ratings
        input_ratings = input_candidates[input_candidates['candidateId'].\
                            isin(group['candidateId'].tolist())]
        # Make a rating list with **only** candidates present in both groups
        # to perform the pearson correlation
        group_ratings = group[group.candidateId.isin(
            input_ratings.candidateId.tolist())]
        group_ratings = group_ratings['score'].tolist()
        input_ratings = input_ratings['score'].tolist()

        """
        # DEBUG:
        temp_df = group_ratings.merge(ir,
                                      left_on='candidateId',
                                      right_on='candidateId',
                                      how='inner')
        print(temp_df)
        # Let the lists have the same length
        group_len = len(group_ratings)
        rating_len = len(input_ratings)

        if group_len > rating_len:
            print('Not same length')
            group_ratings = group_ratings[:rating_len]
        if rating_len > group_len:
            print('Not same length')
            input_ratings = input_ratings[:group_len]
        """
        # Make the correlation
        if len(input_ratings) > 3:
            correlation_factor, _ = stats.pearsonr(input_ratings, group_ratings)
        else:
            # If the lists have a few elements correlation cannot be determined
            correlation_factor= 0

        pearson_correlation[company] = correlation_factor

    return pearson_correlation

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

def make_input_list(max_num_employees, candidates, max_score):

    input_candidates = []

    size = np.random.randint(max_num_employees)
    candidates_to_rank = np.random.randint(candidates, size=size)
    # Get a list of unique candidates
    candidates_to_rank = list(set(candidates_to_rank))

    for candidate in candidates_to_rank:
        score = np.random.randint(low=1, high=max_score+1)
        input_candidates.append([candidate, score])

    input_candidates = pd.DataFrame(input_candidates,
                                    columns=['candidateId', 'score'])
    return input_candidates
