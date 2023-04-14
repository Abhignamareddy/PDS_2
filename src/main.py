# -*- coding: utf-8 -*-
# importing necessary libraries
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

# setting seed for reproducibility
np.random.seed(123)

# reading the data file
data = pd.read_csv("../data_raw/diabetes.csv")

# taking a random sample of 25 observations
sample = data.sample(n=25)

# finding the mean and highest glucose values of the sample
sample_mean_glucose = sample['Glucose'].mean()
sample_highest_glucose = sample['Glucose'].max()

# finding the mean and highest glucose values of the population
population_mean_glucose = data['Glucose'].mean()
population_highest_glucose = data['Glucose'].max()

# creating a histogram of glucose values for the population and the sample
plt.hist(data['Glucose'], alpha=0.5, label='Population')
plt.hist(sample['Glucose'], alpha=0.5, label='Sample')
plt.axvline(population_mean_glucose, color='blue', linestyle='dashed', linewidth=2, label='Population Mean')
plt.axvline(sample_mean_glucose, color='red', linestyle='dashed', linewidth=2, label='Sample Mean')
plt.legend()
plt.xlabel('Glucose')
plt.ylabel('Frequency')
plt.title('Histogram of Glucose Values')
plt.savefig('../results/result1.png')
plt.show()

# creating a scatter plot of glucose values against patient IDs for the population and the sample
plt.scatter(data.index, data['Glucose'], alpha=0.5, label='Population')
plt.scatter(sample.index, sample['Glucose'], alpha=0.5, label='Sample')
plt.axhline(population_highest_glucose, color='blue', linestyle='dashed', linewidth=2, label='Population Highest')
plt.axhline(sample_highest_glucose, color='red', linestyle='dashed', linewidth=2, label='Sample Highest')
plt.legend()
plt.xlabel('Patient IDs')
plt.ylabel('Glucose')
plt.title('Scatter Plot of Glucose Values')
plt.savefig('../results/result2.png')
plt.show()

"""The above code first imports the necessary libraries such as pandas, numpy, random, and matplotlib. It then sets a seed for reproducibility using numpy's random.seed() function.

The code then reads the diabetes.csv file using pandas' read_csv() function and stores it in a variable called data. It then takes a random sample of 25 observations from the population using pandas' sample() function and stores it in a variable called sample.

Next, the code finds the mean and highest glucose values of the sample and the population using pandas' mean() and max() functions.

The code then creates two charts - a histogram of glucose values and a scatter plot of glucose values against patient IDs. For the histogram, the code uses matplotlib's hist() function to plot the glucose values of the population and the sample. It also adds vertical lines to the histogram to show the mean glucose values of the population and the sample. For the scatter plot, the code uses matplotlib's scatter() function to plot the glucose values of the population and the sample against patient IDs. It also adds horizontal lines to the scatter plot to show the highest glucose values of the population and the sample.

Finally, the code displays the two charts using matplotlib's show() function.
"""


# Load the diabetes dataset as a pandas dataframe
df = pd.read_csv("../data_raw/diabetes.csv")

# Set seed for reproducibility
np.random.seed(42)

# Take a random sample of 25 observations
sample = df.sample(25)

# Calculate the 98th percentile of BMI for the population and the sample
pop_percentile = np.percentile(df['BMI'], 98)
sample_percentile = np.percentile(sample['BMI'], 98)

# Print the results
print("Population 98th percentile of BMI:", pop_percentile)
print("Sample 98th percentile of BMI:", sample_percentile)

# Create a box plot to visualize the distribution of BMI in the population and the sample
plt.boxplot([df['BMI'], sample['BMI']])
plt.xticks([1, 2], ['Population', 'Sample'])
plt.ylabel('BMI')
plt.title('Comparison of BMI distribution between population and sample')
plt.savefig('../results/result3.png')
plt.show()

"""This code first loads the diabetes dataset as a pandas dataframe. It then sets a seed for reproducibility, takes a random sample of 25 observations, and calculates the 98th percentile of BMI for both the population and the sample. Finally, it creates a box plot to visualize the distribution of BMI in the population and the sample.

The output of this code will print the 98th percentile of BMI for both the population and the sample. Additionally, it will show a box plot comparing the distribution of BMI between the population and the sample. The box plot will have two boxes side by side, one representing the population and one representing the sample, with the y-axis showing BMI values.
"""

# Set seed for reproducibility
np.random.seed(42)

# Create empty arrays to store the mean, standard deviation, and 50th percentile of each bootstrap sample
mean_arr = np.zeros(500)
std_arr = np.zeros(500)
percentile_arr = np.zeros(500)

# Create 500 bootstrap samples of 150 observations each and calculate the mean, standard deviation, and 50th percentile of BloodPressure for each sample
for i in range(500):
    sample = df['BloodPressure'].sample(n=150, replace=True)
    mean_arr[i] = sample.mean()
    std_arr[i] = sample.std()
    percentile_arr[i] = np.percentile(sample, 50)

# Calculate the mean, standard deviation, and 50th percentile of BloodPressure for the population
pop_mean = df['BloodPressure'].mean()
pop_std = df['BloodPressure'].std()
pop_percentile = np.percentile(df['BloodPressure'], 50)

# Print the results
print("Population mean of BloodPressure:", pop_mean)
print("Average bootstrap mean of BloodPressure:", mean_arr.mean())
print("Population standard deviation of BloodPressure:", pop_std)
print("Average bootstrap standard deviation of BloodPressure:", std_arr.mean())
print("Population 50th percentile of BloodPressure:", pop_percentile)
print("Average bootstrap 50th percentile of BloodPressure:", percentile_arr.mean())

# Create histograms to visualize the distribution of the mean, standard deviation, and 50th percentile of BloodPressure for the bootstrap samples
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].hist(mean_arr)
axs[0].set_xlabel('Mean')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Distribution of mean of BloodPressure')
axs[1].hist(std_arr)
axs[1].set_xlabel('Standard deviation')
axs[1].set_ylabel('Frequency')
axs[1].set_title('Distribution of standard deviation of BloodPressure')
axs[2].hist(percentile_arr)
axs[2].set_xlabel('50th percentile')
axs[2].set_ylabel('Frequency')
axs[2].set_title('Distribution of 50th percentile of BloodPressure')
plt.savefig('../results/result4.png')
plt.show()

"""This code first loads the diabetes dataset as a pandas dataframe. It then sets a seed for reproducibility and creates empty arrays to store the mean, standard deviation, and 50th percentile of BloodPressure for each bootstrap sample. It then creates 500 bootstrap samples of 150 observations each and calculates the mean, standard deviation, and 50th percentile of BloodPressure for each sample. Finally, it calculates the mean, standard deviation, and 50th percentile of BloodPressure for the population and creates histograms to visualize the distribution of the mean, standard deviation, and 50th percentile of BloodPressure for the bootstrap samples.

The output of this code will print the mean, standard deviation, and 50th percentile of BloodPressure for both the population and the average bootstrap samples. Additionally, it will show histograms comparing the distribution of the mean, standard deviation, and 50th percentile of BloodPressure between the population and the bootstrap samples
"""
