import argparse
from collections import Counter
import glob
import json
import os

import gensim
import MeCab
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
import torch

from vocab import Vocabulary
import net


def read_data(data_dir):
    files = glob.glob(f'{data_dir}/*/*.txt')
    categories = []
    titles = []
    for fname in files:
        elems = fname.split('/')
        category = elems[-2]
        with open(fname) as f:
            next(f)
            next(f)
            title = next(f).rstrip()
        categories.append(category)
        titles.append(title)
    print('#Files:', len(titles))
    return categories, titles


def create_dataset(args, categories, titles):
    categories_train, categories_test, titles_train, titles_test = train_test_split(
        categories,
        titles,
        random_state=42,
        stratify=categories,
    )

    dist = Counter(categories_train)
    print('Training')
    for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(v, k)
    print()

    dist = Counter(categories_test)
    print('Test')
    for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True):
        print(v, k)

    category_vocab = Vocabulary()
    feature_vocab = Vocabulary()

    feature_vocab.add_item('<s>')
    feature_vocab.add_item('</s>')
    feature_vocab.add_item('<unk>')
    feature_vocab.add_item('<pad>')

    # Create vocabularies
    for category in categories_train:
        if not category in category_vocab:
            category_vocab.add_item(category)

    tagger = MeCab.Tagger('-Owakati')

    word_freq = Counter()
    titles_tokenized_train = []
    for title in titles_train:
        result = tagger.parse(title).rstrip()
        words = result.split()
        word_freq.update(words)
        titles_tokenized_train.append(words)

    for word, freq in sorted(
        word_freq.most_common(args.vocab_size),
        key=lambda x: x[1],
        reverse=True
    ):
        feature_vocab.add_item(word)

    unk = feature_vocab.get_index('<unk>')
    def create_idx_data(cs, ts):
        y = []
        X = []
        for category in cs:
            y_i = category_vocab.get_index(category)
            y.append(y_i)

        for title_tokenized in ts:
            x = []
            for word in title_tokenized:
                if word in feature_vocab:
                    word_idx = feature_vocab.get_index(word)
                else:
                    word_idx = unk
                x.append(word_idx)
            X.append(x)
        assert len(y) == len(X), f'{len(y)} != {len(x)}'
        return y, X

    y_train, X_train = create_idx_data(categories_train, titles_tokenized_train)

    titles_tokenized_test = []
    for title in titles_test:
        result = tagger.parse(title).rstrip()
        words = result.split()
        titles_tokenized_test.append(words)
    y_test, X_test = create_idx_data(categories_test, titles_tokenized_test)
    print('#Train:', len(y_train), '#Test:', len(y_test))

    category_vocab.save('category.dict')
    feature_vocab.save('feature.dict')

    if args.embedding_path:
        kv = gensim.models.KeyedVectors.load_word2vec_format(args.embedding_path)
        kv.save('embedding.bin')

    torch.save([
        {
            'label': y,
            'words': torch.tensor(x, dtype=torch.long),
            'raw_words': z,
        }
        for y, x, z in zip(y_train, X_train, titles_tokenized_train)
    ], 'train.pt')
    torch.save([
        {
            'label': y,
            'words': torch.tensor(x, dtype=torch.long),
            'raw_words': z,
        }
        for y, x, z in zip(y_test, X_test, titles_tokenized_test)
    ], 'test.pt')


def preprocess(args):
    categories, titles = read_data(args.data_dir)
    create_dataset(args, categories, titles)


def collate_fn(pad):
    def _collate_fn(samples):
        batch = {}
        for sample in sorted(samples, key=lambda x: len(x['words']), reverse=True):
            for k, v in sample.items():
                if not k in batch:
                    batch[k] = [v]
                else:
                    batch[k].append(v)
        batch['label'] = torch.tensor(batch['label'], dtype=torch.long)
        batch['words'] = torch.nn.utils.rnn.pad_sequence(
            batch['words'],
            batch_first=True,
            padding_value=pad
        )
        return batch
    return _collate_fn


def move_to_cuda(batch, gpu):
    batch['label'] = batch['label'].cuda(gpu)
    batch['words'] = batch['words'].cuda(gpu)
    return batch


def train(args):
    with open(args.param_file, 'w') as f:
        param = vars(args)
        del param['handler']
        json.dump(param, f, indent=4)

    feature_vocab = Vocabulary.load('feature.dict')
    category_vocab = Vocabulary.load('category.dict')

    data = torch.load('train.pt')

    pad = feature_vocab.get_index('<pad>')

    model = net.Classifier(
        feature_vocab,
        category_vocab,
        embedding_size=args.embedding_size,
        embedding_path=args.embedding_path,
        freeze_embedding=args.freeze_embedding,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        weight_dropout=args.weight_dropout)

    if args.gpu >= 0:
        model.cuda(args.gpu)
    print(model)

    optimizer = torch.optim.AdamW(model.parameters())
    print(optimizer)

    model.train()
    optimizer.zero_grad()

    for epoch in range(args.max_epochs):
        loss_epoch = 0.
        step = 0
        for batch in torch.utils.data.DataLoader(
            data,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn(pad),
        ):
            optimizer.zero_grad()

            if args.gpu >= 0:
                batch = move_to_cuda(batch, args.gpu)

            loss = net.loss_fn(model, batch)

            loss.backward()
            loss_epoch += loss.item()
            del loss

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            optimizer.step()

            step += 1
        print(f'epoch:{epoch+1}: loss:{loss_epoch:.5f}')
    torch.save(model.state_dict(), args.model)
    evaluate(args)


def evaluate(args):
    feature_vocab = Vocabulary.load('feature.dict')
    category_vocab = Vocabulary.load('category.dict')

    with open(args.param_file, 'r') as f:
        params = json.load(f)

    model = net.Classifier(
        feature_vocab,
        category_vocab,
        **params)

    model.load_state_dict(torch.load(args.model))
    if args.gpu >= 0:
        model = model.cuda(args.gpu)

    test_data = torch.load('test.pt')
    predictions = []
    targets = []
    model.eval()
    pad = feature_vocab.get_index('<pad>')
    match = 0
    with torch.no_grad():
        for batch in torch.utils.data.DataLoader(
            test_data,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn(pad),
        ):

            if args.gpu >= 0:
                batch = move_to_cuda(batch, args.gpu)

            pred = torch.argmax(model(batch), dim=-1)
            target = batch['label']

            match += (pred == target).sum().item()

            predictions.extend(pred.tolist())
            targets.extend(target.tolist())

    acc = match / len(targets)
    prec, rec, fscore, _ = precision_recall_fscore_support(predictions, targets)
    print('Acc', acc)
    print('===')
    print('Category', 'Precision', 'Recall', 'Fscore', sep='\t')
    for idx in range(len(category_vocab)):
        print(f'{category_vocab.get_item(idx)}\t'
              f'{prec[idx]:.2f}\t{rec[idx]:.2f}\t{fscore[idx]:.2f}')
    prec, rec, fscore, _ = precision_recall_fscore_support(predictions, targets, average='micro')
    print(f'Total\t{prec:.2f}\t{rec:.2f}\t{fscore:.2f}')



def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.add_argument('--vocab-size', default=32000, type=int)
    parser_preprocess.add_argument('-d', dest='data_dir', default='text')
    parser_preprocess.add_argument('--embedding-path')
    parser_preprocess.set_defaults(handler=preprocess)

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--batch-size', default=16, type=int)
    parser_train.add_argument('--clip', default=0.1, type=float)
    parser_train.add_argument('--gpu', default=-1, type=int)
    parser_train.add_argument('--embedding-size', default=100, type=int)
    parser_train.add_argument('--embedding-path')
    parser_train.add_argument('--freeze-embedding', action='store_true')
    parser_train.add_argument('--hidden-size', default=128, type=int)
    parser_train.add_argument('--max-epochs', default=3, type=int)
    parser_train.add_argument('--model', default='model.pt')
    parser_train.add_argument('--num-layers', default=1, type=int)
    parser_train.add_argument('--param-file', default='param.json')
    parser_train.add_argument('--weight-dropout', default=0.1, type=float)
    parser_train.set_defaults(handler=train)

    parser_evaluate = subparsers.add_parser('evaluate')
    parser_evaluate.add_argument('--batch-size', default=16, type=int)
    parser_evaluate.add_argument('--gpu', default=-1, type=int)
    parser_evaluate.add_argument('--param-file', default='param.json')
    parser_evaluate.add_argument('--model', default='model.pt')
    parser_evaluate.set_defaults(handler=evaluate)

    args = parser.parse_args()

    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        parser.print_help()




if __name__ == '__main__':
    main()
