from django.shortcuts import render
from django.http import HttpResponse
from django.urls import reverse
from django.shortcuts import redirect
from transformers import T5ForConditionalGeneration, T5Tokenizer
import PyPDF2
import torch
import string
from nltk.corpus import stopwords
from flashtext import KeywordProcessor
import pke
from nltk.tokenize import sent_tokenize
from similarity.normalized_levenshtein import NormalizedLevenshtein
import numpy as np
from sense2vec import Sense2Vec
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.corpus import wordnet as wn
import nltk
import random
import spacy

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('brown')
nltk.download('wordnet')

summary_model = T5ForConditionalGeneration.from_pretrained('t5-base')
summary_tokenizer = T5Tokenizer.from_pretrained('t5-base')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model = summary_model.to(device)

question_model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_squad_v1')
question_model = question_model.to(device)

normalized_levenshtein = NormalizedLevenshtein()
s2v = Sense2Vec().from_disk('s2v_old')

sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

def generate_lessons(paragraphs):
    lessons = []

    nlp = spacy.load("en_core_web_sm")
    s2v = Sense2Vec().from_disk('../../s2v_old')
    sentence_transformer_model = SentenceTransformer('msmarco-distilbert-base-v3')

    for paragraph in paragraphs:

        important_keywords = get_keywords(paragraph, summarizer(paragraph, summary_model, summary_tokenizer))

        lesson = {
            'paragraph': paragraph,
            'keywords': [],
            'questions': []
        }

        for keyword in important_keywords:
            keyword_info = {'keyword': keyword, 'explanation': ''}

            doc = nlp(keyword)
            for token in doc:
                if token.pos_ in ['NOUN', 'PROPN']:
                    synsets = wn.synsets(token.text, pos=token.pos_.lower())
                    if synsets:
                        keyword_info['explanation'] = f"The term '{token.text}' is a {token.pos_.lower()} and refers to {synsets[0].definition()}."

            sense = s2v.get_best_sense(keyword, senses=["NOUN", "VERB", "ADJ", "ADV"])
            most_similar = s2v.most_similar(sense, n=5)
            keyword_info['sense2vec_explanation'] = f"Using Sense2Vec, '{keyword}' is related to: {', '.join([w[0].split('|')[0] for w in most_similar])}."

            embedding = sentence_transformer_model.encode([keyword])
            similar_keywords = sentence_transformer_model.most_similar(embedding, paragraphs, top_k=3)
            keyword_info['sentence_transformer_explanation'] = f"Using Sentence Transformer, '{keyword}' is similar to: {', '.join(similar_keywords)}."

            lesson['keywords'].append(keyword_info)

            questions = []
            for _ in range(4):
                question = get_question(paragraph, keyword, question_model, question_tokenizer)
                distractors = get_distractors(keyword, paragraph, s2v, sentence_transformer_model, 40, 0.2)[:3]
                selected_distractors = random.sample(distractors, min(3, len(distractors)))
                questions.append({'question': question, 'distractors': selected_distractors, 'answer': keyword})

            lesson['questions'].extend(questions)

        lessons.append(lesson)

    return lessons

def get_nouns_multipartite(content):
    out = []
    try:
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=content, language='en')
        pos = {'PROPN', 'NOUN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=15)

        for val in keyphrases:
            out.append(val[0])
    except Exception as e:
        out = []
        print(f"An error occurred in get_nouns_multipartite: {e}")

    return out

def split_into_paragraphs(text):
    paragraphs = text.split('\n')
    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]

def summarizer(text, model, tokenizer):
    text1 = text.strip().replace("\n", " ")
    text1 = "summarize: " + text1
    max_len = 1024
    encoding = tokenizer.encode_plus(text1, max_length=max_len, pad_to_max_length=False, truncation=True,
                                     return_tensors="pt").to(device)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids,
                          attention_mask=attention_mask,
                          early_stopping=True,
                          num_beams=3,
                          num_return_sequences=1,
                          no_repeat_ngram_size=2,
                          min_length=75,
                          max_length=300)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
    summary = dec[0]
    summary = postprocesstext(summary)
    summary = summary.strip()

    return summary

def postprocesstext(content):
    final = ""
    for sent in sent_tokenize(content):
        sent = sent.capitalize()
        final = final + " " + sent
    return final

def get_keywords(originaltext, summarytext):
    keywords = get_nouns_multipartite(originaltext)
    keyword_processor = KeywordProcessor()
    for keyword in keywords:
        keyword_processor.add_keyword(keyword)

    keywords_found = keyword_processor.extract_keywords(summarytext)
    keywords_found = list(set(keywords_found))

    important_keywords = []
    for keyword in keywords:
        if keyword in keywords_found:
            important_keywords.append(keyword)

    return important_keywords[:4]

def get_question(context, answer, model, tokenizer):
    text = "context: {} answer: {}".format(context, answer)
    encoding = tokenizer.encode_plus(text, max_length=384, pad_to_max_length=False, truncation=True, return_tensors="pt").to(device)
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    outs = model.generate(input_ids=input_ids, attention_mask=attention_mask, early_stopping=True, num_beams=5,
                          num_return_sequences=1, no_repeat_ngram_size=2, max_length=72)

    dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]

    Question = dec[0].replace("question:", "")
    Question = Question.strip()
    return Question

def sense2vec_get_words(word, s2v, topn, question):
    output = []
    try:
        sense = s2v.get_best_sense(word, senses=["NOUN", "PERSON", "PRODUCT", "LOC", "ORG", "EVENT", "NORP", "WORK OF ART", "FAC", "GPE", "NUM", "FACILITY"])
        most_similar = s2v.most_similar(sense, n=topn)
        output = filter_same_sense_words(sense, most_similar)
        print("Similar ", output)
    except:
        output = []

    threshold = 0.6
    final = [word]
    checklist = question.split()
    for x in output:
        if get_highest_similarity_score(final, x) < threshold and x not in final and x not in checklist:
            final.append(x)

    return final[1:]

def get_distractors_wordnet(word):
    distractors = []
    try:
        syn = wn.synsets(word, 'n')[0]

        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
    except:
        print("Wordnet distractors not found")
    return distractors

def get_distractors(answer, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    distractors = sense2vec_get_words(answer, sense2vecmodel, top_n, origsentence)
    if len(distractors) == 0:
        return distractors

    distractors_new = [answer.lower()] 
    distractors_new.extend(d.lower() for d in distractors)

    embedding_sentence = origsentence + " " + answer.lower()
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)

    final = [answer.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != answer.lower():
            answer_words = set(answer.lower().split())
            distractor_words = set(wrd.lower().split())
            if not answer_words.intersection(distractor_words):
                final.append(wrd.capitalize())
    final = final[1:]
    return final

def filter_same_sense_words(original, wordlist):
    filtered_words = []
    base_sense = original.split('|')[1]
    print(base_sense)
    for eachword in wordlist:
        if eachword[0].split('|')[1] == base_sense:
            filtered_words.append(eachword[0].split('|')[0].replace("_", " ").title().strip())
    return filtered_words

def get_highest_similarity_score(wordlist, wrd):
    score = []
    for each in wordlist:
        score.append(normalized_levenshtein.similarity(each.lower(), wrd.lower()))
    return max(score)

def mmr(doc_embedding, word_embeddings, words, top_n, lambda_param):
    word_doc_similarity = cosine_similarity(word_embeddings, doc_embedding)
    word_similarity = cosine_similarity(word_embeddings)

    keywords_idx = [np.argmax(word_doc_similarity)]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        mmr = (lambda_param) * candidate_similarities - (1 - lambda_param) * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        keywords_idx.append(mmr_idx)
        candidates_idx.remove(mmr_idx)

    return [words[idx] for idx in keywords_idx]

def main():
    try:
        with open("something.txt", "r", encoding="utf-8") as file:
            text = file.read()

        paragraphs = split_into_paragraphs(text)

        for idx, paragraph in enumerate(paragraphs, start=1):
            print(f"\n--- Paragraph {idx} ---\n")
            print(paragraph)

            text_to_summarize = " ".join(paragraph.split('\n')[:4])
            summarized_text = summarizer(text_to_summarize, summary_model, summary_tokenizer)
            important_keywords = get_keywords(paragraph, summarized_text)
            print(f"{important_keywords}")

            questions_answers = [(answer, get_question(text_to_summarize, answer, question_model, question_tokenizer))
                                 for answer in important_keywords]

            for answer, question in questions_answers:
                print(f"{answer}: {question}")

            choices = {}
            print("Q/A: ", questions_answers)
            for answer, question in questions_answers:
                answer = answer.capitalize()
                distractors = get_distractors(answer, summarized_text, s2v, sentence_transformer_model, 40, 0.2)[:3]
                selected_distractors = random.sample(distractors, min(3, len(distractors)))
                print("distractors:", distractors)
                choices[question] = [answer] + selected_distractors
                random.shuffle(choices[question])
                print("Choices: ", choices[question])

            lessons = generate_lessons([paragraph])

            for lesson in lessons:
                print(f"\nLesson for Paragraph {idx} - Keyword Explanations:")
                for keyword_info in lesson['keywords']:
                    print(f"\nKeyword: {keyword_info['keyword']}")
                    print(f"Explanation: {keyword_info['explanation']}")
                    print(f"Sense2Vec Explanation: {keyword_info['sense2vec_explanation']}")
                    print(f"Sentence Transformer Explanation: {keyword_info['sentence_transformer_explanation']}")

                print("\nQuestions:")
                for question_info in lesson['questions']:
                    print(f"\nQuestion: {question_info['question']}")
                    print(f"Distractors: {question_info['distractors']}")
                    print(f"Answer: {question_info['answer']}")

    except Exception as e:
        print(f"Error processing text: {e}")

if __name__ == "__main__":
    main()
