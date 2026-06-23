# AI Data Sources for TL;DR + Smart Highlights

This document describes the datasets available for training and evaluating the AI summarization and highlight extraction system.

## Quick Start

```bash
# Install dependencies
pip install datasets pandas scikit-learn

# Prepare training data from multiple sources
python data_loaders.py --output-dir ./data --datasets reddit_tldr samsum

# Train classifier
python train_classifier.py --data-path data/train.csv --output models/classifier.pth
```

## Available Datasets

### 1. Reddit TLDR Datasets

#### Webis-TLDR-17
- **Size**: ~3M post-summary pairs
- **Source**: Hugging Face `webis/tldr-17`
- **Best for**: Social media summarization, forum-style text
- **Format**: Reddit posts with user-written TL;DR summaries

```python
from data_loaders import ForumDataLoader

loader = ForumDataLoader()
dataset = loader.load_reddit_tldr('webis-tldr-17')
```

#### TLDRHQ (Reddit TIFU)
- **Size**: ~120K examples
- **Source**: Hugging Face `reddit_tifu`
- **Best for**: High-quality extreme summarization
- **Format**: Reddit posts from r/tifu with TL;DR

```python
dataset = loader.load_reddit_tldr('tldrhq')
```

### 2. Dialogue Summarization

#### SAMSum
- **Size**: ~16K dialogues
- **Source**: Hugging Face `samsum`
- **Best for**: Conversational summarization, chat threads
- **Format**: Messenger-style dialogues with manual summaries

```python
dataset = loader.load_samsum()
```

### 3. News Summarization (Transfer Learning)

#### CNN/DailyMail
- **Size**: ~300K articles
- **Source**: Hugging Face `cnn_dailymail`
- **Best for**: Pretraining, abstractive summarization
- **Format**: News articles with bullet-point summaries

```python
dataset = loader.load_cnn_dailymail()
```

#### XSum
- **Size**: ~200K articles
- **Source**: Hugging Face `xsum`
- **Best for**: Extreme summarization (one sentence)
- **Format**: BBC articles with single-sentence summaries

```python
dataset = loader.load_xsum()
```

## Dataset Comparison

| Dataset | Size | Domain | Compression | Best Use |
|---------|------|--------|-------------|----------|
| Webis-TLDR-17 | 3M | Reddit/Social | High | Forum summarization |
| TLDRHQ | 120K | Reddit | Very High | Extreme summarization |
| SAMSum | 16K | Chat | Medium | Conversation threads |
| CNN/DM | 300K | News | Medium | Transfer learning |
| XSum | 200K | News | Very High | Single-line summaries |

## Creating Sentence Classification Data

The data loader can automatically create sentence-level classification datasets for highlight extraction:

```python
from data_loaders import ForumDataLoader

loader = ForumDataLoader()

# Load thread data
threads = loader.load_reddit_tldr('tldrhq')

# Create sentence classification dataset
dataset = loader.create_sentence_classification_dataset(
    threads=[dict(t) for t in threads],
    output_path='data/sentence_classification.csv'
)
```

Output format:
```csv
sentence,category
"The bug is caused by...",fact
"Try updating to...",solution
"I think this is better",opinion
"How do I fix this?",question
"See https://...",citation
"Thanks for help!",irrelevant
```

## Mixing Datasets

You can mix multiple datasets with custom sampling weights:

```python
reddit_data = loader.load_reddit_tldr('tldrhq')
samsum_data = loader.load_samsum()
cnn_data = loader.load_cnn_dailymail()

# Mix with 50% Reddit, 30% SAMSum, 20% CNN
mixed_dataset = loader.merge_datasets(
    [reddit_data, samsum_data, cnn_data],
    sample_weights=[0.5, 0.3, 0.2]
)
```

## Data Preprocessing

### Automatic Sentence Categorization

The loader includes heuristic categorization for creating training labels:

- **fact**: Sentences from summaries, declarative statements
- **solution**: Contains action words (try, use, should, fix)
- **opinion**: Contains opinion markers (I think, IMO, IMHO)
- **question**: Ends with '?' or starts with question words
- **citation**: Contains URLs or source references
- **irrelevant**: Greetings, thanks, off-topic

### Custom Data Format

Load your own forum data:

```python
# JSON format
threads = loader.load_custom_forum_data('mydata.json', format='json')

# JSONL format (one thread per line)
threads = loader.load_custom_forum_data('mydata.jsonl', format='jsonl')

# CSV format
threads = loader.load_custom_forum_data('mydata.csv', format='csv')
```

Expected format:
```json
{
  "thread_id": "123",
  "content": "Full thread text...",
  "summary": "TL;DR summary...",
  "metadata": {
    "author": "user123",
    "score": 42
  }
}
```

## Training Pipeline

### Full Pipeline Example

```bash
# 1. Prepare data from multiple sources
python data_loaders.py \
  --output-dir ./data \
  --datasets reddit_tldr samsum cnn_dailymail

# Output:
# - data/train.csv (sentence classification data)
# - data/val.csv
# - data/test.csv
# - data/threads.jsonl (full thread data)

# 2. Train dendritic classifier
python train_classifier.py \
  --data-path data/train.csv \
  --output models/classifier.pth \
  --batch-size 32 \
  --lr 2e-5

# 3. Evaluate on test set
python evaluate_classifier.py \
  --model models/classifier.pth \
  --test-data data/test.csv
```

## Data Statistics

After running `prepare_training_data()`, you'll get:

```json
{
  "train": 50000,
  "val": 10000,
  "test": 10000,
  "total_threads": 35000,
  "category_distribution": {
    "fact": 15000,
    "solution": 12000,
    "opinion": 8000,
    "question": 7000,
    "citation": 5000,
    "irrelevant": 3000
  }
}
```

## Dataset Licenses

- **Webis-TLDR-17**: Creative Commons
- **Reddit TIFU**: Reddit API Terms
- **SAMSum**: Non-commercial research
- **CNN/DailyMail**: Research use
- **XSum**: Non-commercial research

Always check individual dataset licenses before commercial use.

## Advanced: Custom Labeling

For better quality, manually label a subset:

```python
# Export unlabeled sentences for annotation
sentences = loader._split_sentences(thread['content'])

# Save for labeling tool
with open('to_label.json', 'w') as f:
    json.dump([{'id': i, 'text': s} for i, s in enumerate(sentences)], f)

# After labeling, merge with auto-labeled data
labeled_df = pd.read_csv('labeled.csv')
auto_df = pd.read_csv('data/train.csv')

# Prefer labeled data where available
final_df = pd.concat([labeled_df, auto_df]).drop_duplicates(subset=['sentence'], keep='first')
```

## Performance Tips

1. **Start small**: Begin with SAMSum (16K) for fast iteration
2. **Add domain data**: Mix in Reddit TLDR for forum-specific patterns
3. **Transfer learning**: Pretrain on CNN/DM, fine-tune on forum data
4. **Balance classes**: Use weighted sampling for imbalanced categories
5. **Quality over quantity**: 10K high-quality labels > 100K noisy ones

## Next Steps

1. Run data preparation to create training datasets
2. Train the dendritic classifier
3. Evaluate on test set
4. Deploy to production API
5. Collect real user data for continuous improvement

For questions or issues, see the main README or open an issue.
