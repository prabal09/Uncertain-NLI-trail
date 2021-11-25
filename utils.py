import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dataloader import get_dfs
import numpy as np
di = dict()
di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"
df_train,df_dev,df_test = get_dfs(di)
def get_df_snli(path = "snli_1.0_train.txt",save_csv = False):
    df = pd.read_csv(path, sep="\t", usecols = ['gold_label','sentence1', 'sentence2'])
#    df['gold_label_num'] = df.gold_label.apply(to_rating)
#    df.head()
    if save_csv:
        if path[-5]=='n':
            name = 'snli_train.csv'
        elif path[-5]=='t':
            name = 'snli_test.csv'
        else:
            name = 'snli_dev.csv'
        df.to_csv(name)
    return df

def showTokenCount(df):
    word_count_pre = df['pre'].apply(lambda x: len(x.split()))
    word_count_hyp = df['hyp'].apply(lambda x: len(x.split()))
    print(word_count_hyp)
    fig = plt.figure(figsize=[10,7])
    # sns.histplot(word_count, color=sns.xkcd_rgb['greenish teal'],fill=False)
    sns.distplot(word_count_pre, color=sns.xkcd_rgb['greenish teal'],label='Premise')
    sns.distplot(word_count_hyp,label='Hypothesis')
    sns.distplot(word_count_hyp + word_count_pre,label='Total')
    plt.xlabel('Tokens')
    plt.ylabel('Density')
    plt.title('Pre-Hyp token count-density')
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()


#df = pd.read_csv("snli_1.0_train.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2']) #train
#df = pd.read_csv("snli_1.0_test.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2'])  #test
#df = pd.read_csv("snli_1.0_dev.txt", sep="\t", usecols = ['gold_label','sentence1', 'sentence2'])  #dev
def to_rating(rating):
  if rating == 'neutral':
      return 1
  elif rating == 'entailment':
      return 0
  else: 
      return 2

#df['gold_label_num'] = df.gold_label.apply(to_rating)
#df.head()
#df.to_csv('snli_train.csv')
#df.to_csv('snli_test.csv')
#df.to_csv('snli_dev.csv')

di = dict()
di["usnli_train"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/train.csv"
di["usnli_dev"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/dev.csv"
di["usnli_test"] = "C:/Users/praba/PycharmProjects/UncertainNLI/u-snli/test.csv"

df_train = pd.read_csv(di['usnli_train'])
df_dev = pd.read_csv(di['usnli_dev'])
df_test = pd.read_csv(di['usnli_test'])

def avg_uNLI(df,inf = 'ENT'):
    if inf == 'ENT':
        df_inf = df[df['nli']==2]
    elif inf == 'NEU':
        df_inf = df[df['nli']==1]
    else:
        df_inf = df[df['nli']==0]
    return np.mean(df_inf.unli)

def snli2unli(di,path_snli = "snli_1.0_train.txt",type_ = 'train'):
    df_snli = get_df_snli(path_snli)
    name = 'usnli_' + type_
    df_unli = pd.read_csv(di[name])
    avg_ent = avg_uNLI(df_unli,'ENT')
    avg_neu = avg_uNLI(df_unli,'NEU')
    avg_con = avg_uNLI(df_unli,'CON')
    df_snli_u = df_snli
    df_snli_u['unli'] = 0
    values = [avg_neu,avg_ent,avg_con]
    conditions = [(df_snli_u['gold_label'] == 'neutral'),
                  (df_snli_u['gold_label'] == 'entailment'),
                  (df_snli_u['gold_label'] == 'contradiction')]
    df_snli_u['unli'] = np.select(conditions, values)
    return df_snli_u

if __name__ == "__main__":
    print('Average uNLI for ENT ',avg_uNLI(df_train,'ENT'))
    print('Average uNLI for CON ',avg_uNLI(df_train,'CON'))
    print('Average uNLI for NEU ',avg_uNLI(df_train,'NEU'))