
import pandas as pd
import matplotlib.pyplot as plt

#losses_df = pd.read_csv('loss_lin_v1.csv',usecols=[1]) # usecols=
#accuracy_df = pd.read_csv('accuracy_lin_v1.csv',usecols=[1]) # usecols=
#losses_df = pd.read_csv('96_74/loss.csv',usecols=[1]) # usecols=
#accuracy_df = pd.read_csv('96_74/accuracy.csv',usecols=[1]) # usecols=
losses_df = pd.read_csv('tt/loss.csv') # usecols=
accuracy_df = pd.read_csv('tt/accuracy.csv') # usecols=

losses_df.plot(title="losses", xlabel= "epochs" , ylabel= "loss")
plt.xlim(right=80)
plt.legend('', frameon=False)
plt.show()
accuracy_df.plot(title="accuracies", xlabel= "epochs", ylabel= "accuracy")
plt.xlim(right=80)
plt.legend('', frameon=False)
plt.show()