import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('train.csv')
print(df.columns)
df['labels']= df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].apply(lambda x: ','.join(x.dropna().astype(str)),axis=1)
print(df.labels.unique())
print(df.info())
df.drop(['id','toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], axis=1, inplace=True)

df_bert_train, df_bert_dev = train_test_split(df, test_size=0.1)
print(df_bert_train.info())
print(df_bert_dev.info())
df_bert_train.to_csv('./dataset/train.tsv', sep='\t', index=False, header=False)
df_bert_dev.to_csv('./dataset/dev.tsv', sep='\t', index=False, header=False)
