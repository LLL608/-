# %% [markdown]

"""
Homework:

The folder '~//data//homework' contains a folder 'Data', containing hand-digits of letters a-z stored in .txt.

Try to establish a network to classify the digits.

`dataLoader.py` offers APIs for loading data.
"""
# %%
import dataLoader as dl

features,labels=dl.readData(r'../data/homework/Data')

# %%
import matplotlib.pyplot as plt
plt.plot(features[5,0:30],features[5,30:])
plt.suptitle="Real: "+labels[5]
plt.show()
# %%
# feature engineering (if necessary)
# %%
# train-test split
# %%
# build the network
# %%
# training
# %%
# predict and evaluate