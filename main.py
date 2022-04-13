import pandas as pd

output = pd.read_csv("data/sample_submission.csv")

test_data = pd.read_json("data/test.json")
train_data = pd.read_json("data/train.json")
pd.set_option('display.max_columns', 20 )
pd.set_option('display.max_rows' , 40)

relevant_train_data = train_data.drop(['end_time_seconds_youtube_clip', 'start_time_seconds_youtube_clip'], axis=1)
print(relevant_train_data.head())
