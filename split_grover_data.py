import json
import os

for f in os.listdir('data'):
    if f.startswith('generator'):
        with open('data/'+f) as infile:
            data = [json.loads(elt) for elt in infile.readlines()]
        
        for i in range(len(data)):
            data[i]['text'] = data[i]['article']
            del data[i]['article']

        train_data = [elt for elt in data if elt['split'] == 'train']
        val_data = [elt for elt in data if elt['split'] == 'val']
        test_data = [elt for elt in data if elt['split'] == 'test']

        for data_split, mode in zip([train_data, val_data, test_data], ['train', 'test', 'valid']):
            with open('data/'+f.replace('.jsonl', f'.{mode}.jsonl'), 'w') as outfile:
                for elt in data_split:
                    outfile.write(json.dumps(elt)+'\n')