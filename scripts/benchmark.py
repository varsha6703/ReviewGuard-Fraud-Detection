import time, statistics
from app.schemas import ReviewInput
from app.inference import predict_review

def main():
    samples=[("A useful product that works well.",1,4) for _ in range(100)]
    durations=[]
    for text,velocity,rating in samples:
        start=time.perf_counter(); predict_review(text,velocity,rating); durations.append((time.perf_counter()-start)*1000)
    print({"requests":len(durations),"avg_ms":round(statistics.mean(durations),3),
           "p95_ms":round(sorted(durations)[int(.95*(len(durations)-1))],3),
           "note":"Record hardware and whether model or fallback was used."})
if __name__=="__main__": main()
