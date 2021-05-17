from transformers import AutoTokenizer

def main(args):
    args.save_dir = util.get_save_dir(args.save_dir, args.name, training=True)
    log = util.get_logger(args.save_dir, args.name)
    tbx = SummaryWriter(args.save_dir)
    device, args.gpu_ids = util.get_available_devices()
    log.info(f'Args: {dumps(vars(args), indent=4, sort_keys=True)}')
    args.batch_size *= max(1, len(args.gpu_ids))

    # Set random seed
    log.info(f'Using random seed {args.seed}...')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Get embeddings
    log.info('Loading embeddings...')
    word_vectors = util.torch_from_json(args.word_emb_file)
    log.info('Building model...')
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

    # if torch.cuda.is_available():
    #   print("------------------------------")
    # device = torch.device("cuda")
    # else:
    device = torch.device("cpu")
    model.to(device)

    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    # print(train_dataset)
    training_args = TrainingArguments("test_trainer", evaluation_strategy="epoch")
    train_dataset = transformers.data.processors.squad.SquadV2Processor().get_train_examples("./data")
    eval_dataset = transformers.data.processors.squad.SquadV2Processor().get_dev_examples("./data")
    # print(train_dataset)

    raw_datasets = {"train": train_dataset, "test": eval_dataset, "unsupervised": None}

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

    small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
    small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))
    full_train_dataset = tokenized_datasets["train"]
    full_eval_dataset = tokenized_datasets["test"]

    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_dataset, eval_dataset=eval_dataset
    )
