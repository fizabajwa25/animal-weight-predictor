# Fiza Bajwa

# this python script creates a fake animal dataset and trains a machine learning model
# (linear regression) to predict an animal's weight based on its height and length.
# it uses randomly generated data for six species: cat, dog, panda, giraffe, gorilla, and zebra.
# after training the model, it makes a prediction for a new animal and shows a scatter plot
# to visualize how body dimensions relate to weight.

import pandas as pd                      # for creating and handling data tables
import numpy as np                      # for numerical operations and arrays
from sklearn.linear_model import LinearRegression  # for creating a linear regression model
import matplotlib.pyplot as plt         # for creating graphs and plots
import random                           # for randomly selecting animal species

# set a random seed so results are the same every time the script runs
np.random.seed(42)

# list of animals to use in the dataset
species = ['Cat', 'Dog', 'Panda', 'Giraffe', 'Gorilla', 'Zebra']

# create a dictionary with randomly generated animal data
data = {
    # randomly assign one of the species to each row (50 total)
    'Species': [random.choice(species) for _ in range(50)],

    # generate 50 random height values between 40 cm and 500 cm
    'Height_cm': np.random.uniform(40, 500, 50).round(1),

    # generate 50 random length values between 60 cm and 350 cm
    'Length_cm': np.random.uniform(60, 350, 50).round(1),
}

# calculate weight based on simple formula using height and length
# add some random noise to make the data more realistic
data['Weight_kg'] = (
    0.25 * np.array(data['Height_cm']) + 
    0.35 * np.array(data['Length_cm']) + 
    np.random.normal(0, 10, 50)          # adds variation so it's not a perfect line
).round(1)

# convert the dictionary to a pandas dataframe 
df = pd.DataFrame(data)

# select the height and length columns as our input (features)
X = df[['Height_cm', 'Length_cm']]

# select weight as the value we want to predict (target)
y = df['Weight_kg']

# create a linear regression model
model = LinearRegression()

# train the model using our data
model.fit(X, y)

# make a prediction for a new animal with height = 160 cm and length = 220 cm
new_animal = pd.DataFrame({'Height_cm': [160], 'Length_cm': [220]})
predicted_weight = model.predict(new_animal)

# print the predicted weight
print("Predicted weight (kg):", round(predicted_weight[0], 2))

# plot a graph to show the relationship between height/length and weight
plt.scatter(df['Height_cm'], df['Weight_kg'], label='Height vs Weight')
plt.scatter(df['Length_cm'], df['Weight_kg'], label='Length vs Weight')
plt.xlabel("Body Dimension (cm)")       # x-axis label
plt.ylabel("Weight (kg)")               # y-axis label
plt.title("Animal Weight vs Body Size") # chart title
plt.legend()                            # add labels to the dots
plt.grid(True)                          # add gridlines to the chart
plt.show()                              # display the chart