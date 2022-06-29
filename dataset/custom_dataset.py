import os
import json
import random
from PIL import Image
from torch.utils.data import Dataset
from dataset.utils import pre_question
import pandas as pd
import numpy as np


class travlr_dataset(Dataset):
    def __init__(self, index_file, transform, vqa_root, eos='[SEP]', split="train",
                 max_ques_words=50, no_caption=True, no_image=False, factor=1, dropout=False, 
                 image_size=384, image_background_color=(255, 255, 255), random_background_shades=False):
        df = pd.read_csv(vqa_root + index_file, index_col=0)
        df["id"] = df.index
        if no_caption:
            df["Q"] = df['query'].apply(lambda x: ' '.format(x))
        else:
            df["Q"] = df['query']
        
        df = self.select_subset(df, factor)

        pd.set_option('max_colwidth', None)

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.dataset = df[(df['split'] == split)]

        self.id_list = self.dataset.id.tolist()
        self.caption_list = self.dataset.caption.tolist()
        self.question_list = self.dataset.Q.tolist()

        findings = list(df.columns[-2:])
        self.findings_list = findings
        self.answer_list = self.dataset.answer.tolist()
        self.image_list = self.dataset.image.tolist()
        self.no_image = no_image
        self.random_background_shades = random_background_shades
        self.image_background_color = image_background_color
        self.image_size = image_size
    
        if dropout:
            self.dropout_arr = random.choices(
                [0, 1, 2], [0.5, 0.25, 0.25], k=len(self.id_list))
        else:
            self.dropout_arr = None

    def modify_query(self, df):
        def add_to_the(string):
            words = string.split()
            words = list(map(lambda x: "to the " +
                         x if x in ['left', 'right'] else x, words))
            return ' '.join(words)
        df['query'] = df['query'].apply(add_to_the)
        return df

    def select_subset(self, df, factor, test_factor=1):
        train = df[df.split == "train"]
        train = train.sample(len(train.index)//factor, random_state=1)
        print(len(train.index))

        val = df[df.split == "val"]
        val = val.sample(len(val.index)//test_factor, random_state=1)
        print(len(val.index))

        test = df[df.split == "test"]
        test = test.sample(len(test.index)//test_factor, random_state=1)
        print(len(test.index))
        df = pd.concat([train, val, test])
        return df

    def convert_to_true_false_task(self, df):
        one_hot = pd.get_dummies(df['answer'])
        df = df.drop('answer', axis=1)
        df = df.join(one_hot)
        df = df.rename(columns={False: "False", True: "True"})
        options = list(df.columns[-2:])
        print(options)

        def fillMask(row):
            real_label = options[0] if row[options[0]] == 1 else options[1]
            wrong_label = options[0] if row[options[0]] == 0 else options[1]
            true = random.randint(0, 1)
            if true:
                newtxt = row["TXT"].replace("[MASK]", real_label)
            else:
                newtxt = row["TXT"].replace("[MASK]", wrong_label)
            return pd.Series([newtxt, true], index=['TXT', 'answer'])
        df[["TXT", "answer"]] = df.apply(fillMask, axis=1)
        return df

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        target = self.answer_list[index]
        caption = self.caption_list[index]
        question = self.question_list[index]
        image = self.image_list[index]
        dropout_caption = self.dropout_arr is not None and self.dropout_arr[index] == 1
        if dropout_caption:
            caption = None
        dropout_image = self.dropout_arr is not None and self.dropout_arr[index] == 2

        if self.random_background_shades:
            rand_num = random.randint(0, 255)
            background_color = (rand_num, rand_num, rand_num)
        else:
            background_color = self.image_background_color
        
        if self.no_image or dropout_image:
            # image_path = os.path.join("/home/kengjich/blank/images/-1.jpg")
            
            image = Image.new('RGB',(self.image_size, self.image_size), background_color)
        else:
            image_path = os.path.join(self.vqa_root, "images/" + str(image) + ".jpg")
            image = Image.open(image_path).convert('RGB')
            image_data = image.load()
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if image_data[i, j] == (255, 255, 255): # The original dataset is while background
                        image_data[i, j] = background_color
                        
        image = self.transform(image)
        text = [caption, question]

        # question = pre_question(caption,self.max_ques_words)
        return image, text, int(target)


class custom_dataset(Dataset):
    def __init__(self, index_file, transform, vqa_root, eos='[SEP]', split="train",
                 max_ques_words=50, no_caption=True, no_image=False, factor=1, dropout=False, 
                 image_size=384, image_background_color=(255, 255, 255), random_background_shades=False):
        df = pd.read_csv(vqa_root + index_file, index_col=0)
        df["id"] = df.index
        if no_caption:
            df["TXT"] = df['query'].apply(lambda x: '[SEP] {}'.format(x))
        else:
            df["TXT"] = df[['caption', 'query']].apply(
                lambda x: '{} [SEP] {}'.format(x[0], x[1]), axis=1)  # add caption
        df = self.select_subset(df, factor)

        pd.set_option('max_colwidth', None)
        print(df)
        print(df["TXT"])

        self.transform = transform
        self.vqa_root = vqa_root
        self.max_ques_words = max_ques_words
        self.eos = eos

        self.dataset = df[(df['split'] == split)]

        self.id_list = self.dataset.id.tolist()
        self.report_list = self.dataset.TXT.tolist()

        findings = list(df.columns[-2:])
        self.findings_list = findings
        self.answer_list = self.dataset.answer.tolist()
        self.image_list = self.dataset.image.tolist()
        self.no_image = no_image
        self.random_background_shades = random_background_shades
        self.image_background_color = image_background_color
        self.image_size = image_size
    
        if dropout:
            self.dropout_arr = random.choices(
                [0, 1, 2], [0.5, 0.25, 0.25], k=len(self.id_list))
        else:
            self.dropout_arr = None

    def modify_query(self, df):
        def add_to_the(string):
            words = string.split()
            words = list(map(lambda x: "to the " +
                         x if x in ['left', 'right'] else x, words))
            return ' '.join(words)
        df['query'] = df['query'].apply(add_to_the)
        return df

    def select_subset(self, df, factor, test_factor=1):
        train = df[df.split == "train"]
        train = train.sample(len(train.index)//factor, random_state=1)
        print(len(train.index))

        val = df[df.split == "val"]
        val = val.sample(len(val.index)//test_factor, random_state=1)
        print(len(val.index))

        test = df[df.split == "test"]
        test = test.sample(len(test.index)//test_factor, random_state=1)
        print(len(test.index))
        df = pd.concat([train, val, test])
        return df

    def convert_to_true_false_task(self, df):
        one_hot = pd.get_dummies(df['answer'])
        df = df.drop('answer', axis=1)
        df = df.join(one_hot)
        df = df.rename(columns={False: "False", True: "True"})
        options = list(df.columns[-2:])
        print(options)

        def fillMask(row):
            real_label = options[0] if row[options[0]] == 1 else options[1]
            wrong_label = options[0] if row[options[0]] == 0 else options[1]
            true = random.randint(0, 1)
            if true:
                newtxt = row["TXT"].replace("[MASK]", real_label)
            else:
                newtxt = row["TXT"].replace("[MASK]", wrong_label)
            return pd.Series([newtxt, true], index=['TXT', 'answer'])
        df[["TXT", "answer"]] = df.apply(fillMask, axis=1)
        return df

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        target = self.answer_list[index]
        caption = self.report_list[index]
        image = self.image_list[index]
        dropout_caption = self.dropout_arr is not None and self.dropout_arr[index] == 1
        if dropout_caption:
            caption = "[SEP]" + caption.split("[SEP]")[1]
        dropout_image = self.dropout_arr is not None and self.dropout_arr[index] == 2

        if self.random_background_shades:
            rand_num = random.randint(0, 255)
            background_color = (rand_num, rand_num, rand_num)
        else:
            background_color = self.image_background_color
        
        if self.no_image or dropout_image:
            # image_path = os.path.join("/home/kengjich/blank/images/-1.jpg")
            
            image = Image.new('RGB',(self.image_size, self.image_size), background_color)
        else:
            image_path = os.path.join(self.vqa_root, "images/" + str(image) + ".jpg")
            image = Image.open(image_path).convert('RGB')
            image_data = image.load()
            for i in range(self.image_size):
                for j in range(self.image_size):
                    if image_data[i, j] == (255, 255, 255): # The original dataset is while background
                        image_data[i, j] = background_color
                        
        image = self.transform(image)

        question = caption
        # question = pre_question(caption,self.max_ques_words)
        return image, question, int(target)