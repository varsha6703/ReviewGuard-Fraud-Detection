import os, random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from faker import Faker

def main():
    os.makedirs("data", exist_ok=True)
    fake=Faker()
    random.seed(42); np.random.seed(42)
    n_reviews=50_000; fraud_rate=.05; start=datetime(2024,1,1)
    data=[]
    for i in range(n_reviews):
        user_id=random.randint(1,10_000)
        post_date=start+timedelta(days=random.randint(0,365))
        velocity=int(np.random.poisson(2))
        is_fraud=random.random()<fraud_rate
        if is_fraud:
            sentence=fake.sentence(nb_words=8, ext_word_list=["amazing","best ever","5 stars"])
            text=" ".join([sentence]*random.randint(2,4))
            rating=5; velocity=int(np.random.poisson(8))
        else:
            text=fake.paragraph(nb_sentences=3)
            rating=random.randint(1,5)
        data.append({"review_id":i+1,"user_id":user_id,"text":text,"rating":rating,
                     "velocity":velocity,"timestamp":post_date.isoformat(),"is_fraud":int(is_fraud)})
    df=pd.DataFrame(data)
    df.to_csv("data/reviews_dataset.csv",index=False)
    print(f"Generated {len(df)} reviews; fraud={int(df.is_fraud.sum())}; legitimate={int((df.is_fraud==0).sum())}")
if __name__=="__main__": main()
