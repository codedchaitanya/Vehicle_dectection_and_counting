import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
data_path = "../Downloads/gtdump.xmp"
df = pd.read_csv(data_path)
hr_wave = df.iloc[:,3]
time = df.iloc[:,0]/1000
norm_wave = (hr_wave - np.mean(hr_wave))/np.std(hr_wave)
print(df.iloc[:,3])
plt.plot(time[:600],norm_wave[:600])
plt.show()
