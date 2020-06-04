import pandas as pd

def get_speaker_counts(df):
    client_ids = list(df.client_id.unique())
    
    sentence_counts = {}

    for i in range(len(client_ids)):
        client_id = client_ids[i]
        count = len(df[df['client_id'] == client_id])
        sentence_counts[client_id] = count

    sorted_customer_ids = [k for k, v in sorted(sentence_counts.items(), key=lambda cust: cust[1], reverse=True)]
    
    return sentence_counts, sorted_customer_ids


def build_speaker_dataset(df, top_n_speakers=3):
    sentence_counts, sorted_customer_ids = get_speaker_counts(df)
    
    client_id = sorted_customer_ids[0]
    training_df = df[df['client_id'] == client_id]

    if top_n_speakers >= len(sorted_customer_ids):
        raise ValueError("Number of speakers requested is larger than the total number of unique speakers")
    
    for client_id_index in range(1, top_n_speakers):
        client_id = sorted_customer_ids[client_id_index]
        training_df = pd.concat([training_df, df[df['client_id'] == client_id]])

    return training_df