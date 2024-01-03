import numpy as np
import pandas as pd

import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from transformers import set_seed, pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, BertTokenizer
from nltk.tokenize import sent_tokenize
import torch

from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainer, Seq2SeqTrainingArguments

from datasets import Dataset, DatasetDict

from datasets import load_metric
from tqdm import tqdm

orig_df = pd.read_csv("youtube_reviews.csv")

headphone_names = pd.Series(orig_df['Headphone_Name'].unique())

engadget = ['Sony nearly did it again. The company has dominated both over-ear and true wireless product categories for the last few years. It has a knack for creating a compelling combination of sound quality, noise cancelling performance, customization and features. None of the competition comes close to what the WF-1000XM4 offers in terms of what the earbuds can do for you automatically with features like Adaptive Sound Control and Speak-to-Chat. These are almost the complete package, if only the new ear tips offered a better fit. Even the best of the three pairs included in the box never felt truly comfortable. I only found relief when I grabbed the silicone tips from the M3 instead, and most people won’t have access to those. It seems so simple, but if you mess it up, a basic thing like ear tips can nearly ruin otherwise stellar earbuds. The WF-1000XM4 is available now in black and silver color options for $280.',
            'I’ve said a set of Samsung’s Galaxy Buds are its best yet before – more than once. That’s because the company continues to improve its formula with each subsequent release, whether that’s the regular Buds or the Buds Pro. And now I have to declare it again. The Buds 2 Pro are a huge leap from the 2021 Pro model, with massive improvements to the audio, notable gains in noise cancellation and the introduction of several new features. Samsung lets its loyal customers unlock the best of the Buds 2 Pro, the same way Apple and Google have done. That’s not likely to change, but Samsung is making a strong case for owners of its phones to invest in its audio products too.',
            '',
            'If it’s supreme noise blocking you’re looking for in your next set of true wireless earbuds, the QCE II is the choice. With the updates Bose delivers here with the help of CustomTune, not only is the ANC noticeably better than the previous model, but overall audio quality and ambient sound mode are also improved. Sure, I’d like more than six hours of battery life and conveniences like multipoint connectivity and wireless charging should be standard fare at this point. For $299, I’d expect some of those basics to be included and Bose passed on them.',
            'Bose has come a long way since the SoundSport Free. The company had years to perfect its next set(s) of true wireless earbuds, and it’s created a tempting package. The QuietComfort Earbuds have powerful ANC and great overall sound quality, plus premium features like wireless charging. The limited customization and touch controls could be a headache for some, and the large-sized buds create a look some may not want. And when you factor in price, Sony’s WF-1000XM3 is an attractive alternative despite its age. Bose and Sony have done battle over noise-cancelling headphones during the last few years, now they’re doing the same for true wireless earbuds. And Bose finally has a product that can give Sony a run for its money.',
            'If you’re looking for the best of what AirPods has to offer in earbuds that don’t have the polarizing stick apparatus, the Beats Fit Pro should do the trick. They offer a nice blend of features, sound and noise-cancelling performance for the price. Sure, there are better options but they also cost significantly more, especially if you’re looking for the absolute best audio quality. For now, Beats is giving the masses an AirPods alternative that’s actually still packed with Apple tech. And that’s an interesting proposition for iPhone owners.',
            'Apple’s noise-canceling earbuds were way overdue for an update. While the company didn’t see the need to change the overall design, it did extensive upgrades on the inside, introducing new features and improving performance along the way. Importantly, it made all of these changes while keeping the price at $249. Things like improved audio, more powerful ANC, Adaptive Transparency and even the upgrades to the charging case make the new AirPods Pro a worthwhile update to a familiar formula. Let’s just hope we don’t have to wait another three years for a full redesign.',
            'Sony largely succeeded at what it set out to do: It built a set of true wireless earbuds that offers transparent audio by design rather than relying on microphones to pipe in ambient sound. Indeed, the LinkBuds blend your music, podcasts or videos with whatever is going on around you. There are certainly benefits for this, whether it be the ability to be less of a jerk in the office or to stay safe outdoors. Even with all of the handy tech Sony packs in, earbuds need to be comfortable enough to wear for long periods of time, and the area around the unique ring-shaped drivers is simply too hard to be accommodating. Consistent audio performance would make a big difference, too. For now, the LinkBuds are an interesting product that could be more compelling with some refinements. Hopefully Sony will do just that, because I’m very much looking forward to version 2.0. The LinkBuds are available to order today from Amazon and Best Buy in grey and white color options for $180.',
            '',
            'Google’s best earbuds yet are also its most complete package thus far. All of the features that made 2020’s redesigned Pixel Buds and the A-Series follow-up such compelling options for Android users, especially Pixel owners, are back. And while the Pixel Buds Pro are $20 more than what we got two years ago, the 2022 version is much improved. Active noise cancellation and the refined sound quality are equally impressive, and well worth the extra money. As long as Google can deliver spatial audio quickly and it works well, the only thing lacking is call quality, which may not be a dealbreaker for you.',
            '',
            '',
            '',
            'With the WF-1000XM5, Sony improves its already formidable mix of great sound, effective ANC and handy features. These earbuds are undoubtedly the company’s best and most comfortable design in its premium model so far, which was one of the few remaining riddles Sony needed to solve. For all of the company’s ability to add so many features, many of them still need fine-tuning, but that doesn’t make them any less useful in their current state. The WF-1000XM5 are more expensive too, which means the competition has one key area it can beat Sony. As is typically the case, there aren’t many flaws with the company’s latest model and its rivals still have their work cut out for them. The WF-1000XM5 are available for pre-order now in black and silver color options for $300. According to Amazon, the earbuds will ship on August 4th.',
            '',
            '']



summaries_df = pd.concat([headphone_names, pd.Series(engadget)], axis = 1)

summaries_df = summaries_df.rename(columns={0: 'Headphone_Name', 1: 'Summary'})

summaries_df = summaries_df.set_index('Headphone_Name')


individual_reviews_df = orig_df.merge(summaries_df, on='Headphone_Name', how='left')

#just removing reviewtext title error
individual_reviews_df = individual_reviews_df.rename(columns={'Sony_Review_Text': 'Review_Text'})


device = "cuda" if torch.cuda.is_available() else "cpu"

model_ckpt = "google/pegasus-cnn_dailymail"
#model_ckpt = "google/pegasus-multi_news"

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt).to(device)

def generate_batch_sized_chunks(list_of_elements, batch_size):
    """split the dataset into smaller batches that we can process simultaneously
    Yield successive batch-sized chunks from list_of_elements.
    
    Yields consecutive chunks from a list.

    Args:
        list_of_elements (List[Any]): The list to be divided into chunks.
        batch_size (int): The size of chunks.

    Yields:
        List[Any]: A chunk from the list of the specified size.
        
    """
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]

def calculate_metric_on_test_ds(dataset, metric, model, tokenizer, 
                               batch_size=2, device=device, 
                               column_text="Review_Text", 
                               column_summary="Summary"):
    """
    Calculates a specified metric on a test dataset.

    Args:
        dataset (Dataset): The dataset to evaluate.
        metric (Metric): The metric to calculate.
        model (nn.Module): The model to evaluate.
        tokenizer (Tokenizer): The tokenizer to use for text processing.
        batch_size (int, optional): The batch size for evaluation.
        device (torch.device, optional): The device to use for computation.
        column_text (str, optional): The name of the text column in the dataset.
        column_summary (str, optional): The name of the summary column in the dataset.

    Returns:
        Dict[str, float]: The calculated metric scores.
    """
    article_batches = list(generate_batch_sized_chunks(dataset[column_text], batch_size))
    target_batches = list(generate_batch_sized_chunks(dataset[column_summary], batch_size))

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches), total=len(article_batches)):
        
        inputs = tokenizer(article_batch, max_length=1024,  truncation=True, 
                        padding="max_length", return_tensors="pt")
        
        summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                         attention_mask=inputs["attention_mask"].to(device), 
                         length_penalty=0.8, num_beams=8, max_length=128)
        ''' parameter for length penalty ensures that the model does not generate sequences that are too long. '''
        
        # Finally, we decode the generated texts, 
        # replace the <n> token, and add the decoded texts with the references to the metric.
        decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                clean_up_tokenization_spaces=True) 
               for s in summaries]      
        
        decoded_summaries = [d.replace("<n>", " ") for d in decoded_summaries]
        
        
        metric.add_batch(predictions=decoded_summaries, references=target_batch)
        
    #  Finally compute and return the ROUGE scores.
    score = metric.compute()
    return score

def convert_examples_to_features(example_batch):
    input_encodings = tokenizer(example_batch['Review_Text'] , max_length = 1024, truncation = True )
    
    with tokenizer.as_target_tokenizer():
        target_encodings = tokenizer(example_batch['Summary'], max_length = 128, truncation = True )
        
    return {
        'input_ids' : input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'labels': target_encodings['input_ids']
    }


from sklearn.model_selection import train_test_split

#replacing empty values with na and then dropping those rows
individual_reviews_df = individual_reviews_df.replace(r'^\s*$', pd.NA, regex=True).dropna()

# Define the proportions for the splits
train_size = 0.6
validation_size = 0.2
test_size = 0.2

# First, split the data into a temporary training set and a temporary test set
train, temp_test = train_test_split(individual_reviews_df, test_size=1 - train_size, random_state=42)

# Then, split the temporary test set into the validation set and the final test set
final_validation, final_test = train_test_split(temp_test, test_size=test_size / (test_size + validation_size), random_state=42)


from sklearn.model_selection import train_test_split

#replacing empty values with na and then dropping those rows
individual_reviews_df = individual_reviews_df.replace(r'^\s*$', pd.NA, regex=True).dropna()

# Define the proportions for the splits
train_size = 0.6
validation_size = 0.2
test_size = 0.2

# First, split the data into a temporary training set and a temporary test set
train, temp_test = train_test_split(individual_reviews_df, test_size=1 - train_size, random_state=42)

# Then, split the temporary test set into the validation set and the final test set
final_validation, final_test = train_test_split(temp_test, test_size=test_size / (test_size + validation_size), random_state=42)

#getting dataset in a form for trainign with hugging face libraries
train_ds = Dataset.from_pandas(train)
validation_ds = Dataset.from_pandas(final_validation)
test_ds = Dataset.from_pandas(final_test)

ds = DatasetDict()

ds['train'] = train_ds
ds['validation'] = validation_ds
ds['test'] = test_ds

model_ckpt = "google/pegasus-cnn_dailymail"

pipe = pipeline('summarization', model = model_ckpt )

tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

rouge_metric = load_metric('rouge')

score = calculate_metric_on_test_ds(ds['test'], rouge_metric, model_pegasus, tokenizer)


rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

print(pd.DataFrame(rouge_dict, index = ['pegasus']))

dataset_dict_pt = ds.map(convert_examples_to_features, batched = True)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model_pegasus)


import accelerate
import transformers
from transformers import TrainingArguments, Trainer

trainer_args = TrainingArguments(
    output_dir='pegasus-individual-reviews', num_train_epochs=2,
    per_device_train_batch_size=8, per_device_eval_batch_size=8,
    logging_steps=8,
    evaluation_strategy='epoch', save_steps=1e6
) 


trainer = Trainer(model=model_pegasus, args=trainer_args,
                  tokenizer=tokenizer, data_collator=data_collator,
                  train_dataset=dataset_dict_pt["train"], 
                  eval_dataset=dataset_dict_pt["validation"])

trainer.train()


rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]

rouge_metric = load_metric('rouge')

score = calculate_metric_on_test_ds(
    ds['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'Review_Text', column_summary= 'Summary'
)

rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )

print(pd.DataFrame(rouge_dict, index = [f'pegasus']))