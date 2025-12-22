import glob
import pandas as pd
import matplotlib.pyplot as plt

# Loop over all .dat files in the current directory
for filename in glob.glob("*.dat"):
    print(filename)
    # Read the data
    df = pd.read_csv(
        filename,
        delim_whitespace=True,
        skiprows=14,
        names=["wavel", "trans"]
    )
    
    # Make the plot
    plt.figure()
    plt.plot(df["wavel"], df["trans"])
    plt.xlabel("wavel")
    plt.ylabel("trans")
    plt.title(filename)
    plt.show()