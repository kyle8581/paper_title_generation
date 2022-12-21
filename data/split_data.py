import json
import random




with open("paper_data.json","r") as f:
    paper_data = json.load(f)

print(len(paper_data))

# total_data = [{"title":k, "abstract":v} for k,v in paper_data.items()]
# random.shuffle(total_data)
# train_set = total_data[:int(0.8*len(total_data))]
# valid_set = total_data[int(0.8*len(total_data)):int(0.9*len(total_data))]
# test_set = total_data[int(0.9*len(total_data)):]

# with open("train.json", "w") as f:
#     json.dump(train_set, f, indent=4)
# with open("valid.json", "w") as f:
#     json.dump(valid_set, f, indent=4)
# with open("test.json", "w") as f:
#     json.dump(test_set, f, indent=4)
    

    

