from abc import ABC

from transformers import T5EncoderModel, AutoTokenizer, AutoModel
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, Adafactor, get_linear_schedule_with_warmup, get_constant_schedule
from accelerate import Accelerator
from torch.utils.data import Dataset
from torch import Tensor
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import json
import faiss
import torch
import os
import time
from utils.io import read_file


# pip install faiss-gpu

class BiDataset(Dataset, ABC):
    def __init__(self, data, corpus, tokenizer, max_doc_len=512, max_q_len=128):
        self.data = data
        self.corpus = corpus
        self.tokenizer = tokenizer
        self.max_doc_len = max_doc_len
        self.max_q_len = max_q_len

    def __getitem__(self, item):
        query, doc_id = self.data[item]
        if isinstance(doc_id, list):
            doc_id = doc_id[0]
        doc = self.corpus[doc_id]
        return (torch.tensor(self.tokenizer.encode(query, truncation=True, max_length=self.max_q_len)),
                torch.tensor(self.tokenizer.encode(doc, truncation=True, max_length=self.max_doc_len)))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        query, doc = zip(*data)
        query = pad_sequence(query, batch_first=True, padding_value=0)
        doc = pad_sequence(doc, batch_first=True, padding_value=0)
        return {
            'query': query,
            'doc': doc,
            'ids': None,
            'aux_ids': None
        }


class TestData(Dataset, ABC):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, item):
        text = self.data[item]
        return torch.tensor(self.tokenizer.encode(text, truncation=True, max_length=self.max_length))

    def __len__(self):
        return len(self.data)

    @staticmethod
    def collate_fn(data):
        input_ids = pad_sequence(data, batch_first=True, padding_value=0)
        return {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(0),
        }


def mean_pooling(model_output, attention_mask, **kwargs):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    x = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return x


def contriever_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def load_beir_qg(short_name='covid', loader=False, **kwargs):
    from data.beir.config import DATA_NAME_TO_QG_DIR, DATA_NAME_TO_DIR
    beir = ['arg', 'touche', 'covid', 'nfc', 'hotpot', 'dbp', 'climate', 'fever', 'scifact', 'scidocs', 'fiqa']
    data_name = DATA_NAME_TO_QG_DIR[short_name]
    corpus_json = [json.loads(line) for line in open(f'data/beir/{DATA_NAME_TO_DIR[short_name]}/corpus.jsonl')]
    corpus = []
    id_to_line_num = {}
    for line in corpus_json:
        id_to_line_num[line['_id']] = len(corpus)
        corpus.append(f"Title: {line['title']}. Text: {line['text']}")
    queries = [json.loads(line) for line in open(f'data/beir_qg/{data_name}/qgen-queries.jsonl')]
    qid_to_query = {line['_id']: line['text'] for line in queries}
    data_json = [line.split() for line in open(f'data/beir_qg/{data_name}/qgen-qrels/train.tsv')][1:]
    data = []
    for line in data_json:
        try:
            query = qid_to_query[line[0]]
            doc_num = id_to_line_num[line[1]]
            data.append([query, doc_num])
        except:
            pass
    if loader:
        dataset = BiDataset(data=data, corpus=corpus, tokenizer=kwargs['tokenizer'], max_doc_len=128, max_q_len=32)
        print('Num of beir', len(dataset))
        data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn,
                                                  batch_size=kwargs['batch_size'], shuffle=True, num_workers=8)
        return data_loader
    return data, corpus


def load_beir(short_name='arg'):
    from data.beir.config import DATA_NAME_TO_DIR
    beir = ['arg', 'touche', 'covid', 'nfc', 'hotpot', 'dbp', 'climate', 'fever', 'scifact', 'scidocs', 'fiqa']
    data_name = DATA_NAME_TO_DIR[short_name]
    print(f'load {short_name} {data_name}')
    corpus_json = [json.loads(line) for line in open(f'data/beir/{data_name}/corpus.jsonl')]
    corpus = []
    id_to_line_num = {}
    for line in corpus_json:
        id_to_line_num[line['_id']] = len(corpus)
        corpus.append(f"Title: {line['title']}. Text: {line['text']}")
    queries = [json.loads(line) for line in open(f'data/beir/{data_name}/queries.jsonl')]
    qid_to_query = {line['_id']: line['text'] for line in queries}
    data_json = [line.split() for line in open(f'data/beir/{data_name}/qrels/test.tsv')][1:]
    from collections import defaultdict
    data_kv = defaultdict(list)
    for line in data_json:
        if int(line[2]) == 0:
            continue
        query = qid_to_query[line[0]]
        try:
            data_kv[query].append([id_to_line_num[line[1]], int(line[2])])
        except KeyError:
            print('KeyError', line[1])
    data = []
    for query, doc_nums in data_kv.items():
        # data.append([query, doc_nums])
        if len(doc_nums) > 0:
            data.append([query, doc_nums])
        else:
            print('Missing', query)
    return data, corpus


def train():
    accelerator = Accelerator()
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    epochs = 10
    batch_size = 60

    save_path = 'out/bi-ms+nfc'

    accelerator.print(save_path)

    model = T5EncoderModel.from_pretrained('t5-base')
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    optimizer = AdamW(model.parameters(), 2e-4)

    data = json.load(open('data/ms320k/train.json'))
    corpus = read_file('data/ms320k/corpus.txt')
    dataset = BiDataset(data=data, corpus=corpus, tokenizer=tokenizer, max_doc_len=128, max_q_len=32)
    accelerator.print(f'data size={len(dataset)}')
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=batch_size,
                                              shuffle=True, num_workers=8)

    beir = ['arg', 'touche', 'covid', 'nfc', 'hotpot', 'dbp', 'climate', 'fever', 'scifact', 'scidocs', 'fiqa']
    beir_data_loader = load_beir_qg('scifact', loader=True, tokenizer=tokenizer, batch_size=batch_size)
    beir_data_loader = accelerator.prepare(beir_data_loader)
    beir_data_loader = iter(beir_data_loader)

    model, optimizer, data_loader = accelerator.prepare(model, optimizer, data_loader)
    scheduler = get_constant_schedule(optimizer)

    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        accelerator.print(f'Training epoch {epoch}')
        accelerator.wait_for_everyone()
        model.train()
        tk0 = tqdm(data_loader, total=len(data_loader))
        loss_report = []
        for ms_batch in tk0:
            beir_batch = next(beir_data_loader)
            for batch in [ms_batch, beir_batch]:
                query = mean_pooling(model(batch['query'], attention_mask=batch['query'].ne(0)), batch['query'].ne(0))
                doc = mean_pooling(model(batch['doc'], attention_mask=batch['doc'].ne(0)), batch['doc'].ne(0))
                target = torch.arange(0, query.size(0), 1, device=query.device, dtype=torch.long)
                logits = torch.matmul(query, doc.transpose(0, 1))
                loss = F.cross_entropy(logits, target)
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


def build_index(collection, shard=True, dim=None, gpu=True):
    t = time.time()
    dim = collection.shape[1] if dim is None else dim
    cpu_index = faiss.index_factory(dim, "Flat", faiss.METRIC_INNER_PRODUCT)
    # cpu_index = faiss.index_factory(dim, 'OPQ32,IVF1024,PQ32')
    if gpu:
        ngpus = faiss.get_num_gpus()
        co = faiss.GpuMultipleClonerOptions()
        co.shard = shard
        gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co=co)
        index = gpu_index
    else:
        index = cpu_index

    # gpu_index.train(xb)
    index.add(collection)
    print(f'build index of {len(collection)} instances, time cost ={time.time() - t}')
    return index


def do_retrieval(xq, index, k=1):
    t = time.time()
    distance, rank = index.search(xq, k)
    print(f'search {len(xq)} queries, time cost ={time.time() - t}')
    return rank, distance


def encode(data, model, tokenizer, batch_size, max_length):
    collection = []
    collect = []
    model = model.cuda()
    model.eval()
    for sentence in tqdm(data):
        collect.append(sentence)
        if len(collect) == batch_size:
            batch = tokenizer(collect, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.no_grad():
                model_output = model(**batch)

                # sentence_embeddings = average_pool(model_output.last_hidden_state, batch['attention_mask'])
                # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

                # sentence_embeddings = mean_pooling(model_output, batch['attention_mask'])
                # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
                sentence_embeddings = model_output.pooler_output

                sentence_embeddings = sentence_embeddings.cpu().tolist()
            collection.extend(sentence_embeddings)
            collect = []
    if len(collect) > 0:
        batch = tokenizer(collect, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
        batch = {k: v.cuda() for k, v in batch.items()}
        with torch.no_grad():
            model_output = model(**batch)

            # sentence_embeddings = average_pool(model_output.last_hidden_state, batch['attention_mask'])
            # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

            # sentence_embeddings = mean_pooling(model_output, batch['attention_mask'])
            # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            sentence_embeddings = model_output.pooler_output

            sentence_embeddings = sentence_embeddings.cpu().tolist()
        collection.extend(sentence_embeddings)
    collection = np.array(collection, dtype=np.float32)
    return collection


def loader_encode(data_loader, model):
    collection = []
    model = model.cuda()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = {k: v.cuda() for k, v in batch.items()}
            model_output = model(**batch)
            # sentence_embeddings = mean_pooling(model_output, batch['attention_mask'])
            # sentence_embeddings = model_output.pooler_output
            sentence_embeddings = contriever_pooling(model_output[0], batch['attention_mask'])

            sentence_embeddings = sentence_embeddings.cpu().tolist()
            collection.extend(sentence_embeddings)
    collection = np.array(collection, dtype=np.float32)
    return collection


def test():
    batch_size = 128
    # save_path = 'out/bt5-2'
    # save_path = 'out/e5-large'

    # data = json.load(open('data/new_nq320k/dev.json'))
    # corpus = json.load(open('data/new_nq320k/corpus.json'))
    # corpus = corpus[:512]

    model = T5EncoderModel.from_pretrained('t5-base', ignore_mismatched_sizes=True)
    tokenizer = AutoTokenizer.from_pretrained('t5-base')

    from data.beir.config import DATA_NAME_TO_DIR

    beir = ['arg', 'touche', 'covid', 'nfc', 'hotpot', 'dbp', 'climate', 'fever', 'scifact', 'scidocs', 'fiqa']

    short_name = 'fiqa'

    data_name = DATA_NAME_TO_DIR[short_name]
    save_path = f'out/bt5-{short_name}'

    print(short_name, data_name, save_path)

    corpus_json = [json.loads(line) for line in open(f'data/beir/{data_name}/corpus.jsonl')]
    corpus = []
    id_to_line_num = {}
    for line in corpus_json:
        id_to_line_num[line['_id']] = len(corpus)
        corpus.append(f"Title: {line['title']}. Text: {line['text']}")
    queries = [json.loads(line) for line in open(f'data/beir/{data_name}/queries.jsonl')]
    qid_to_query = {line['_id']: line['text'] for line in queries}
    data_json = [line.split() for line in open(f'data/beir/{data_name}/qrels/test.tsv')][1:]
    from collections import defaultdict
    data_kv = defaultdict(list)
    for line in data_json:
        if int(line[2]) == 0:
            continue
        query = qid_to_query[line[0]]
        try:
            data_kv[query].append([id_to_line_num[line[1]], int(line[2])])
        except KeyError:
            print('KeyError', line[1])
    data = []
    for query, doc_nums in data_kv.items():
        data.append([query, doc_nums])
        # if len(doc_nums) > 0:
        #     data.append([query, doc_nums])
        # else:
        #     print('Missing', query)
    # model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    model = model.cuda()

    for epoch in range(100):
        if os.path.exists(f'{save_path}/{epoch}.pt'):
            model.load_state_dict(torch.load(f'{save_path}/{epoch}.pt'))
        else:
            continue
        print(f'Test {save_path}/{epoch}.pt')
        if os.path.exists(f'{save_path}/{epoch}.pt.rank'):
            rank = json.load(open(f'{save_path}/{epoch}.pt.rank'))
        else:
            collection = encode(corpus, model, tokenizer, batch_size, max_length=128)
            index = build_index(collection, gpu=False)
            query_text = [x[0] for x in data]
            queries = encode(query_text, model, tokenizer, batch_size, max_length=32)
            rank, distance = do_retrieval(queries, index, k=100)
            rank = rank.tolist()

            os.makedirs(save_path, exist_ok=True)
            json.dump(rank, open(f'{save_path}/{epoch}.pt.rank', 'w'))

        query_ids = [x[1] for x in data]
        # metric = [int(res[0] == label) for res, label in zip(rank, query_ids)]
        # print(sum(metric) / len(metric))
        from eval import eval_all
        print(eval_all(rank, query_ids))
        print()


def test_baseline():
    # from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer, \
    #     DPRContextEncoder

    save_path = 'out/contriever'
    print(save_path)
    epoch = 0

    data = json.load(open('data/new_nq320k/dev_unseen.json'))
    corpus = json.load(open('data/new_nq320k/id.newtitle.json'))

    qq = read_file('out/code-002/nq320k.title')
    print(len(data), len(qq))
    # data = [[_x[0] + ' ' + _q.replace('|', ' ').lower(), _x[1]] for _x, _q in zip(data, qq)]
    data = [[_q.replace('|', ' ').lower(), _x[1]] for _x, _q in zip(data, qq)]

    # data = json.load(open('data/ms320k/new_dev.json'))
    # corpus = read_file('data/ms320k/corpus.txt')

    # beir = ['arg', 'touche', 'covid', 'nfc', 'hotpot', 'dbp', 'climate', 'fever', 'scifact', 'scidocs', 'fiqa']
    # data, corpus = load_beir('fiqa')

    tokenizer = AutoTokenizer.from_pretrained('./contriever')
    model = AutoModel.from_pretrained('./contriever')
    model.eval()

    dataset = TestData(corpus, tokenizer, max_length=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=128,
                                              shuffle=False, num_workers=16)
    collection = loader_encode(data_loader, model)

    index = build_index(collection, gpu=False)

    query_text = [x[0] for x in data]
    dataset = TestData(query_text, tokenizer, max_length=128)
    data_loader = torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, batch_size=128,
                                              shuffle=False, num_workers=8)
    queries = loader_encode(data_loader, model)

    rank, distance = do_retrieval(queries, index, k=100)
    rank = rank.tolist()
    os.makedirs(save_path, exist_ok=True)
    json.dump(rank, open(f'{save_path}/{epoch}.pt.rank', 'w'))
    from eval import eval_all
    query_ids = [x[1] for x in data]
    print(eval_all(rank, query_ids))

    # seen_split = json.load(open('data/new_nq320k/dev_seen_split.json'))
    # unseen_split = json.load(open('data/new_nq320k/dev_unseen_split.json'))
    # print('seen:', eval_all([rank[i] for i in seen_split], [query_ids[i] for i in seen_split]))
    # print('unseen:', eval_all([rank[i] for i in unseen_split], [query_ids[i] for i in unseen_split]))


if __name__ == '__main__':
    # do()
    # train()
    # test()
    test_baseline()
