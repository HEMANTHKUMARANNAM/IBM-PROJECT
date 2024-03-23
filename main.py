from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

datainfile = pd.read_csv('UBER.csv')

# vol data
vol =list((datainfile.to_dict()["Volume"]).values())
print(vol)
arry = np.array(vol)

# x axis _ common
a = []
for x in range(0, len(vol)) :
  a.append(x)
arrx = np.array(a)

# plotind x vs vol
plt.figure(figsize=(10,600))
plt.title("DAY X VS VOLUME ")
plt.plot( arrx , arry )
plt.show()

# highest data
high =list((datainfile.to_dict()["High"]).values())
print(high)
arry = np.array(high)

# plotting x vs high values
plt.figure(figsize=(10,600))
plt.title("DAY X VS HIGHEST PRICING ")
plt.plot( arrx , arry  )
plt.show()


# difference arr
diff = []
low =list((datainfile.to_dict()["Low"]).values())
print(low)
arrt = np.array(low)

for x in range(0, len(arrx)):
  diff.append(arry[x] -arrt[x])






# difference of high and low
plt.figure(figsize=(10,600))
plt.title("DAY X VS CHANGE IN RATE ")
plt.plot( arrx , diff  )
plt.show()






# # tree
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn import tree

# from sklearn.model_selection import train_test_split 
# from sklearn.metrics import confusion_matrix, classification_report

# data=pd.read_csv('UBER.csv')

# X=data[['Open','Volume','Low','Close','Adj Close' ]]

# Y=data['High']

# x_train,x_test,y_train,y_test = train_test_split(X,Y, test_size=0.2)



# def modelpred (model, x_train,y_train, x_test): 
#   model.fit(x_train,y_train) 
#   y_pred=model.predict(x_test)
#   tree.plot_tree (model) 
#   plt.show()

#   print(confusion_matrix(y_test,y_pred))

#   print(classification_report(y_test,y_pred))

# ID3=tree.DecisionTreeClassifier (criterion="entropy")
# CART=tree.DecisionTreeClassifier (criterion="gini") 

# modelpred (ID3, x_train, y_train, x_test) 
# modelpred (CART, x_train, y_train, x_test)




# MULTIPLE LINEAR AGGRESSION

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (you should replace this with your own dataset)
data = {
    'Attribute1': vol,
    'Attribute2': list((datainfile.to_dict()["Close"]).values()),
    'Attribute3': diff,
    'Attribute4': list((datainfile.to_dict()["Adj Close"]).values()),
    'Price': high
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Separate the target variable (Price) from the features (attributes)
X = df[['Attribute1', 'Attribute2', 'Attribute3' , 'Attribute4']]
y = df['Price']

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)

# Print the coefficients and intercept of the model
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)



# visual linear

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (replace with your own data)
data = {
    'Attribute1': vol,
    'Attribute2': list((datainfile.to_dict()["Close"]).values()),
    'Attribute3': diff,
    'Attribute4': list((datainfile.to_dict()["Adj Close"]).values()),
    'Price': high,
    'Attribute5':list((datainfile.to_dict()["Low"]).values()) 
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Separate the target variable (Price) from the features (attributes)
X = df[['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5']]
y = df['Price']

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot actual vs. predicted values
plt.scatter(y, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual Prices vs. Predicted Prices (MULTIPLE LINEAR REGRESSION)")

plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)




# ramdom forest regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (replace with your own data)
data = {
    'Attribute1': vol,
    'Attribute2': list((datainfile.to_dict()["Close"]).values()),
    'Attribute3': diff,
    'Attribute4': list((datainfile.to_dict()["Adj Close"]).values()),
    'Price': high,
    'Attribute5':list((datainfile.to_dict()["Low"]).values()) 
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Separate the target variable (Price) from the features (attributes)
X = df[['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5']]
y = df['Price']

# Create a Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Create a scatter plot for one attribute (Attribute1) and y
plt.scatter(df['Attribute1'], y, label='Attribute1 vs. Price', color='blue')
plt.scatter(df['Attribute1'], y_pred, label='Predicted Price', color='red', marker='x')
plt.xlabel("Attribute1")
plt.ylabel("Price")
plt.legend()
plt.title("Attribute1 vs. Actual HIGH Price and Predicted Price ( RANDOM FOREST REGRESSION MODEL)")

# Show the plot
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (replace with your own data)
data = {
    'Attribute1': vol,
    'Attribute2': list((datainfile.to_dict()["Close"]).values()),
    'Attribute3': diff,
    'Attribute4': list((datainfile.to_dict()["Adj Close"]).values()),
    'Price': list((datainfile.to_dict()["Low"]).values()) ,
    'Attribute5': high
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Separate the target variable (Price) from the features (attributes)
X = df[['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5']]
y = df['Price']

# Create a Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Create a scatter plot for one attribute (Attribute1) and y
plt.scatter(df['Attribute1'], y, label='Attribute1 vs. Price', color='blue')
plt.scatter(df['Attribute1'], y_pred, label='Predicted Price', color='red', marker='x')
plt.xlabel("Attribute1")
plt.ylabel("Price")
plt.legend()
plt.title("Attribute1 vs. Actual LOW Price and Predicted Price ( RANDOM FOREST REGRESSION MODEL)")

# Show the plot
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# testing


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Sample data (replace with your own data)
data = {
    'Attribute1': vol,
    'Attribute2': list((datainfile.to_dict()["Close"]).values()),
    'Attribute3': list((datainfile.to_dict()["Low"]).values()),
    'Attribute4': list((datainfile.to_dict()["Volume"]).values()),
    'Price': list((datainfile.to_dict()["Open"]).values()) ,
    'Attribute5': high
}

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Separate the target variable (Price) from the features (attributes)
X = df[['Attribute1', 'Attribute2', 'Attribute3', 'Attribute4', 'Attribute5']]
y = df['Price']

# Create a Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model to the data
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Create a scatter plot for one attribute (Attribute1) and y
plt.scatter(df['Attribute1'], y, label='Attribute1 vs. Price', color='blue')
plt.scatter(df['Attribute1'], y_pred, label='Predicted Price', color='red', marker='x')
plt.xlabel("Attribute1")
plt.ylabel("Price")
plt.legend()
plt.title("Attribute1 vs. Actual OPENING Price and Predicted Price ( RANDOM FOREST REGRESSION MODEL)")

# Show the plot
plt.show()

# Evaluate the model
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


# CURR_VOL_VAL = int(input("ENTER THE VOLUME OF USERS : "))
# y_pred = model.predict([[41.570000,34 ,CURR_VOL_VAL , 35]])
# print(y_pred)