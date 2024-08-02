# Create a summary of the detection per size category
# If this doesn't produce the expected output, check the R code 

import pandas as pd
import scipy.stats as stats

def compute_detection_per_size_ci(data_df, dataset):
    # Read the csv table with the aggregated data
    df = data_df

    # Function to compute mean and 95% confidence interval
    def mean_ci(data):
        mean = data.mean()
        ci = stats.t.interval(0.95, len(data)-1, loc=mean, scale=stats.sem(data))
        return mean, ci

    # Group by size category and apply the function
    agg_results = df.groupby("Size Category").agg({
        "Recall(%)": lambda x: mean_ci(x),
        "FPs per case": lambda x: mean_ci(x)
    })

    # Reformat the results for better readability
    agg_results = pd.DataFrame({
        "Recall(%) Mean": agg_results["Recall(%)"].apply(lambda x: x[0]),
        "Recall(%) CI Lower": agg_results["Recall(%)"].apply(lambda x: x[1][0]),
        "Recall(%) CI Upper": agg_results["Recall(%)"].apply(lambda x: x[1][1]),
        "FPs per case Mean": agg_results["FPs per case"].apply(lambda x: x[0]),
        "FPs per case CI Lower": agg_results["FPs per case"].apply(lambda x: x[1][0]),
        "FPs per case CI Upper": agg_results["FPs per case"].apply(lambda x: x[1][1]),
    })

    #print(agg_results)
    agg_results.to_csv(f'detection_per_size_{dataset}_95CI.csv', index=True)
    print(f"Formatted summary saved to detection_per_size_{dataset}_95CI.csv")


data = 'all_aggregated_MR_detection_per_size.csv'
data_df = pd.read_csv(data)
# Use the function compute_detection_per_size_ci to compute the mean and 95% confidence interval
compute_detection_per_size_ci(data_df, 'MR')