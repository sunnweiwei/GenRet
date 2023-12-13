from abc import ABC

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from accelerate import Accelerator
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import json
from utils.io import write_file, read_file
import os


# train 307373 dev 7830
class NQDataset(Dataset, ABC):
    def __init__(self, data, tokenizer, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        query, doc_id = self.data[item]
        return (torch.tensor(self.tokenizer.encode(str(query), truncation=True, max_length=self.max_len)),
                torch.tensor(self.tokenizer.encode(str(doc_id))))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        inputs, outputs = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return {
            'input_ids': inputs,
            'attention_mask': inputs.ne(0),
            'labels': pad_sequence(outputs, batch_first=True, padding_value=-100),
        }


class NewNQDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_len=128):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, item):
        query, doc_id = self.data[item]
        while isinstance(doc_id, list):
            doc_id = doc_id[0]
        doc = self.corpus[doc_id]
        return (torch.tensor(self.tokenizer.encode(str(query), truncation=True, max_length=self.max_len)),
                torch.tensor(self.tokenizer.encode(str(doc), truncation=True, max_length=self.max_len)))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        inputs, outputs = zip(*data)
        inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        return {
            'input_ids': inputs,
            'attention_mask': inputs.ne(0),
            'labels': pad_sequence(outputs, batch_first=True, padding_value=-100),
        }


class Tree:
    def __init__(self):
        self.root = dict()

    def set(self, path):
        pointer = self.root
        for i in path:
            if i not in pointer:
                pointer[i] = dict()
            pointer = pointer[i]

    def set_all(self, path_list):
        for path in tqdm(path_list):
            self.set(path)

    def find(self, path):
        if isinstance(path, torch.Tensor):
            path = path.cpu().tolist()
        pointer = self.root
        for i in path:
            if i not in pointer:
                return []
            pointer = pointer[i]
        return list(pointer.keys())

    def __call__(self, batch_id, path):
        return self.find(path)


# corpus: "id", 'new_id', '2', "doc", '3', '4', 'en'
# train: "query", "qid", "new_id", "old_id"


def train():
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 100
    batch_size = 128
    save_path = 'out/dsi'
    model = AutoModelForSeq2SeqLM.from_pretrained('models/t5-base')
    tokenizer = AutoTokenizer.from_pretrained('models/t5-base')

    num_of_new_tokens = 10  # 109739

    tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])  # 109739
    model.resize_token_embeddings(len(tokenizer))

    data = json.load(open('dataset/nq320k/train.json'))
    data.extend(json.load(open('dataset/nq320k/qg.json')))

    corpus = json.load(open('dataset/nq320k_id/id.random2.json'))
    corpus = [''.join([f'${i}$' for i in z]) for z in corpus]
    corpus = [f'${z}$' for z in corpus]

    optimizer = AdamW(model.parameters(), 5e-4)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=32)
    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)

    scheduler = get_constant_schedule(optimizer)

    os.makedirs(save_path, exist_ok=True)
    accelerator.print(tokenizer.decode(dataset[128][0]))
    accelerator.print('==>')
    accelerator.print(tokenizer.decode(dataset[128][1]), dataset[128][1])

    for epoch in range(epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for batch in tk0:
            out = model(**batch)
            loss = out.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            loss_report.append(loss.item())
            tk0.set_postfix(loss=sum(loss_report) / len(loss_report))
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            accelerator.save(accelerator.unwrap_model(model).state_dict(), f'{save_path}/{epoch}.pt')


def test():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 1
    save_path = 'out/dsi-ms-title'
    # save_path = 'out/dsi-title'
    num_of_new_tokens = 10  # 109739

    model = AutoModelForSeq2SeqLM.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')


    # tokenizer.add_tokens([f'${i}$' for i in range(num_of_new_tokens)])  # 109739
    # model.resize_token_embeddings(len(tokenizer))

    # tokenizer = AutoTokenizer.from_pretrained("./genre-kilt")
    # model = AutoModelForSeq2SeqLM.from_pretrained("./genre-kilt").eval()

    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    qq = read_file('out/flan-t5-xxl/nq320k.title')
    print(len(data), len(qq))
    data = [[_q.lower(), _x[1]] for _x, _q in zip(data, qq)]

    # corpus = json.load(open('data/new_nq320k/id.bert_km.json'))
    # corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    # data = [[doc, i] for i, doc in enumerate(read_file('data/ms320k/corpus.txt'))]
    # corpus = ['' for _ in range(len(data))]

    # corpus = [f'${z}$' for z in range(109739)]

    # data = json.load(open('data/ms320k/new_dev.json'))
    # corpus = read_file('data/ms320k/id.title.txt')
    # corpus = json.load(open('data/ms320k/id.semantic.json'))
    # corpus = [''.join([f'${i}$' for i in z]) for z in corpus]

    # corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # from run_bi import load_beir
    # from collections import defaultdict
    # data, corpus = load_beir('scidocs')
    # corpus = [' '.join(x.replace('Title: ', '').replace('. Text:', '').strip().split()[:8]).lower() for x in corpus]
    # docid_to_doc = defaultdict(list)
    # for i, item in enumerate(corpus):
    #     docid_to_doc[item].append(i)
    # query_ids = [x[1] for x in data]

    print(len(data), len(corpus))

    corpus_ids = [[0] + tokenizer.encode(line) for line in corpus]
    print(corpus_ids[0])
    tree = Tree()
    tree.set_all(corpus_ids)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()
    seen_split = json.load(open('data/new_nq320k/dev_seen_split.json'))
    unseen_split = json.load(open('data/new_nq320k/dev_unseen_split.json'))
    for epoch in range(10000, 0, -1):
        if not os.path.exists(f'{save_path}/{epoch}.pt'):
            continue
        # print(f'Test {save_path}/{epoch}.pt')
        # model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        label_all = []
        top_k = 1
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=4,
                    num_beams=top_k,
                    num_return_sequences=top_k,
                    length_penalty=None,
                    min_length=None,
                    no_repeat_ngram_size=None,
                    early_stopping=None,
                    prefix_allowed_tokens_fn=tree
                )
                # continue
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
                output = [str(x).replace('$', '').strip() for x in output]
                # print(output)
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line)
                new_output.append(beam)
                # print(len(output))
                batch['labels'][batch['labels'] == -100] = 0
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels = [str(x).replace('$', '').strip() for x in labels]

                acc.extend([int(l in o) for o, l in zip(new_output, labels)])
                tk0.set_postfix(acc=sum(acc) / len(acc))

                # print(new_output[-1], labels[-1])

                print(new_output)
                print(labels)

                output_all.extend(new_output)
                label_all.extend(labels)
        print(f'Test {save_path}/{epoch}.pt, ACC =', sum(acc) / len(acc), end='; ')
        # print('Seen', sum([acc[j] for j in seen_split]) / len(seen_split), end='; ')
        # print('Unseen', sum([acc[j] for j in unseen_split]) / len(unseen_split))
        json.dump([output_all, label_all], open(f'{save_path}/{epoch}.pt.outputs', 'w'))
        from eval import eval_all
        print(eval_all(output_all, label_all))

        break

        # new_predictions = []
        # import copy
        # for line in output_all:
        #     new_line = []
        #     for s in line:
        #         if s not in docid_to_doc:
        #             continue
        #         tmp = copy.deepcopy(docid_to_doc[s])
        #         new_line.extend(tmp)
        #         if len(new_line) > 10:
        #             break
        #     new_predictions.append(new_line)
        # output_all = new_predictions
        #
        # print('BEIR', eval_all(output_all, query_ids))

        # print('Seen')
        # print(eval_all([output_all[j] for j in seen_split], [label_all[j] for j in seen_split]))
        # print('Unseen')
        # print(eval_all([output_all[j] for j in unseen_split], [label_all[j] for j in unseen_split]))


def simple_match():
    import edlib
    import numpy as np
    from thefuzz import fuzz
    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # qq = read_file('out/flan-t5-xxl/nq320k.title')
    qq = read_file('out/code-002/nq320k.title')
    output_all = []
    label_all = []
    metric = []
    for line, item in zip(tqdm(qq), data):
        score = [fuzz.token_sort_ratio(line.lower(), x) for x in corpus]
        # score = [- edlib.align(line.lower(), x)['editDistance'] for x in corpus]
        idx = np.argmax(score)
        output_all.append(corpus[idx])
        label_all.append(corpus[item[1]])
        print(line, corpus[idx], corpus[item[1]], sep=' | ')
        metric.append(idx == item[1])
        print(sum(metric) / len(metric))



def simple_test():
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    batch_size = 10

    model = AutoModelForSeq2SeqLM.from_pretrained('./parrot_paraphraser_t5')
    tokenizer = AutoTokenizer.from_pretrained('./parrot_paraphraser_t5')

    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    # qq = read_file('out/flan-t5-xxl/nq320k.title')
    qq = read_file('out/code-002/nq320k.title')
    print(len(data), len(qq))
    data = [[_q.lower(), _x[1]] for _x, _q in zip(data, qq)]

    print(len(data), len(corpus))

    corpus_ids = [[0] + tokenizer.encode(line) for line in corpus]
    # corpus_ids = [[2, 0] + tokenizer.encode(line)[1:] for line in corpus]
    print(corpus_ids[0])
    tree = Tree()
    tree.set_all(corpus_ids)

    dataset = NewNQDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_len=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=False, num_workers=16)
    model = model.cuda()
    model.eval()
    seen_split = json.load(open('data/new_nq320k/dev_seen_split.json'))
    unseen_split = json.load(open('data/new_nq320k/dev_unseen_split.json'))
    for epoch in range(0, 100):
        # print(f'Test {save_path}/{epoch}.pt')
        # model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
        tk0 = tqdm(data_loader, total=len(data_loader))
        acc = []
        output_all = []
        label_all = []
        top_k = 32
        with torch.no_grad():
            for batch in tk0:
                batch = {k: v.cuda() for k, v in batch.items()}
                output = model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_length=32,
                    num_beams=top_k,
                    num_return_sequences=1,
                    length_penalty=0,
                    no_repeat_ngram_size=0,
                    early_stopping=False,
                    prefix_allowed_tokens_fn=tree,

                    min_length=3,

                    # bos_token_id=0,
                    # decoder_start_token_id=2,
                    # eos_token_id=2,
                    # forced_bos_token_id=0,
                    # forced_eos_token_id=2,
                )
                # print(output)
                # continue
                output = tokenizer.batch_decode(output, skip_special_tokens=True)
                output = [str(x).replace('$', '').strip() for x in output]
                # print(output)
                beam = []
                new_output = []
                for line in output:
                    if len(beam) >= top_k:
                        new_output.append(beam)
                        beam = []
                    beam.append(line)
                new_output.append(beam)
                # print(len(output))
                batch['labels'][batch['labels'] == -100] = 0
                labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                labels = [str(x).replace('$', '').strip() for x in labels]

                acc.extend([int(l in o) for o, l in zip(new_output, labels)])
                tk0.set_postfix(acc=sum(acc) / len(acc))

                # print(new_output[-1], labels[-1])

                print(tokenizer.batch_decode(batch['input_ids'],skip_special_tokens=True), new_output, labels)
                # print(labels)

                output_all.extend(new_output)
                label_all.extend(labels)

        print(f'Test {epoch}.pt, ACC =', sum(acc) / len(acc), end='; ')
        # print('Seen', sum([acc[j] for j in seen_split]) / len(seen_split), end='; ')
        # print('Unseen', sum([acc[j] for j in unseen_split]) / len(unseen_split))
        json.dump([output_all, label_all], open(f'{save_path}/{epoch}.pt.outputs', 'w'))
        from eval import eval_all
        print(eval_all(output_all, label_all))

        break

        # new_predictions = []
        # import copy
        # for line in output_all:
        #     new_line = []
        #     for s in line:
        #         if s not in docid_to_doc:
        #             continue
        #         tmp = copy.deepcopy(docid_to_doc[s])
        #         new_line.extend(tmp)
        #         if len(new_line) > 10:
        #             break
        #     new_predictions.append(new_line)
        # output_all = new_predictions
        #
        # print('BEIR', eval_all(output_all, query_ids))

        # print('Seen')
        # print(eval_all([output_all[j] for j in seen_split], [label_all[j] for j in seen_split]))
        # print('Unseen')
        # print(eval_all([output_all[j] for j in unseen_split], [label_all[j] for j in unseen_split]))


# corpus: "id", '1', '2', "doc", '3', '4', 'en'
# train: "query", "qid", "new_id", "old_id"

def do():
    train_data = json.load(open('data/new_nq320k/train.json'))
    dev_data = json.load(open('data/new_nq320k/dev.json'))
    qg = json.load(open('data/new_nq320k/qg.json'))
    train_id = set([x[1] for x in train_data])
    dev_id = set([x[1] for x in train_data])

    # new_data = [x for x in dev_data if x[1] not in train_id]
    # new_data = [x for x in qg if x[1] not in train_id]
    new_data = [i for i, x in enumerate(dev_data) if x[1] not in train_id]
    # print(len(new_data))
    print(len(new_data))
    # json.dump(new_data, open('data/new_nq320k/dev_seen_split.json', 'w'))  # 6075
    json.dump(new_data, open('data/new_nq320k/dev_unseen_split.json', 'w'))  # 1755

    # json.dump(new_data, open('data/new_nq320k/qg_seen.json', 'w'))  # 1080260
    # json.dump(new_data, open('data/new_nq320k/qg_unseen.json', 'w'))  # 17130


def title_data():
    # "query", "qid", "new_id", "old_id"
    # corpus: "id", '1', '2', "doc", '3', '4', 'en'
    data = [line[:-1].split('\t') for line in open('data/nq320k/train.txt')]
    corpus = {}
    for line in [line[:-1].split('\t') for line in open('data/nq320k/corpus.txt')]:
        old_id, _, _, doc, _, _, _ = line
        corpus[old_id] = doc
    new_data = []
    for line in data:
        query, _, _, old_id = line
        doc = corpus[old_id]
        title = doc.split('  ')[0]
        title = ' '.join(title.split()[:5])
        line.append(title)
        new_data.append('\t'.join(line))
    write_file(new_data, 'data/nq320k/train_title.txt')


def clean_data():
    train_data = [line[:-1].split('\t') for line in open('data/nq_data_sem/nq_train_doc_newid.tsv')]
    old_to_new = dict()
    new_train_data = []
    for line in train_data:
        query, qid, title, new_id, old_id = line
        old_to_new[old_id] = new_id
        new_train_data.append([query, int(old_id)])
    dev_data = [line[:-1].split('\t') for line in open('data/nq_data_sem/nq_dev_doc_newid.tsv')]
    new_dev_data = []
    for line in dev_data:
        query, qid, title, new_id, old_id = line
        old_to_new[old_id] = new_id
        new_dev_data.append([query, int(old_id)])

    corpus = [line[:-1].split('\t') for line in open('data/nq320k/corpus.txt')]
    new_corpus = []
    bert_km = []
    random_id = []
    title_id = []
    for line in corpus:
        old_id, _, _, doc, *_ = line
        # title = doc.split('  ')[0]
        # title = ' '.join(title.split()[:5])
        bert_km.append(str(old_to_new[old_id]))
        # random_id.append(str(old_id).zfill(6))
        # title_id.append(title)
        new_corpus.append(doc)
    # json.dump(new_train_data, open('data/new_nq320k/train.json', 'w'))
    # json.dump(new_dev_data, open('data/new_nq320k/dev.json', 'w'))
    json.dump(new_corpus, open('data/new_nq320k/corpus.json', 'w'))
    # json.dump(bert_km, open('data/new_nq320k/id.simcse.json', 'w'))
    # json.dump(random_id, open('data/new_nq320k/id.random.json', 'w'))
    # json.dump(title_id, open('data/new_nq320k/id.title.json', 'w'))


def tmp():
    from run_bi import load_beir
    from collections import defaultdict
    data, corpus = load_beir('scidocs')
    corpus = [' '.join(x.replace('Title: ', '').replace('. Text:', '').strip().split()[:8]).lower() for x in corpus]
    docid_to_doc = defaultdict(list)
    for i, item in enumerate(corpus):
        docid_to_doc[item].append(i)
    query_ids = [x[1] for x in data]

    from eval import eval_all
    import numpy as np
    prediction = []
    for i in range(len(data)):
        rank = [m for m in range(len(corpus))]
        # np.random.shuffle(rank)
        prediction.append(rank)
    print(eval_all(prediction, query_ids))


if __name__ == '__main__':
    train()
    # tmp()
    # exit()
    # # clean_data()
    # # do()
    # # train()
    # test()
    # simple_test()
    # simple_match()
    # while True:
    #     test()
    # title_data()

