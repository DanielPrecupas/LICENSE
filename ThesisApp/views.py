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

def home(request):
    return render(request, 'main.html')

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

def get_nouns_multipartite(content):
    out = []
    try:
        for paragraph in content:
            extractor = pke.unsupervised.MultipartiteRank()
            extractor.load_document(input=paragraph, language='en')
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

def get_distractors(word, origsentence, sense2vecmodel, sentencemodel, top_n, lambdaval):
    distractors = sense2vec_get_words(word, sense2vecmodel, top_n, origsentence)
    #print("distractors ", distractors)
    if len(distractors) == 0:
        return distractors
    distractors_new = [word.capitalize()]
    distractors_new.extend(distractors)

    embedding_sentence = origsentence + " " + word.capitalize()
    keyword_embedding = sentencemodel.encode([embedding_sentence])
    distractor_embeddings = sentencemodel.encode(distractors_new)

    max_keywords = min(len(distractors_new), 5)
    filtered_keywords = mmr(keyword_embedding, distractor_embeddings, distractors_new, max_keywords, lambdaval)
    final = [word.capitalize()]
    for wrd in filtered_keywords:
        if wrd.lower() != word.lower():
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

def upload_pdf(request):
    if request.method == 'POST' and request.FILES['pdf_file']:
        pdf_file = request.FILES['pdf_file']

        try:
            with pdf_file.open() as pdf:
                pdf_reader = PyPDF2.PdfReader(pdf)
                text = ''
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text()

            print(text)

            sentences = sent_tokenize(text)
            sentences_per_paragraph = 4
            paragraphs = [sentences[i:i + sentences_per_paragraph] for i in range(0, len(sentences), sentences_per_paragraph)]

            print(paragraphs)

            paragraph_questions = {}

            for i, paragraph in enumerate(paragraphs):
                textyes = " ".join(paragraph[:4])
                summarized_text = summarizer(text, summary_model, summary_tokenizer)
                important_keywords = get_keywords(paragraph, textyes)[:3] 

                questions_answers = [(answer, get_question(text, answer, question_model, question_tokenizer))
                                     for answer in important_keywords]
                
            print(paragraph_questions);

            for answer, question in questions_answers:
                print(f"{answer}: {question}")

            choices = {}

            for answer, question in questions_answers:
                answer = answer.capitalize()
                distractors = get_distractors(answer, summarized_text, s2v, sentence_transformer_model, 40, 0.2)[:3]
                selected_distractors = random.sample(distractors, min(3, len(distractors)))
                print("distractors:", distractors)
                choices[question] = [answer] + selected_distractors
                random.shuffle(choices[question])
                print("Choices: ", choices[question])

            paragraph_questions[f'Paragraph {i+1}'] = {'text': paragraph, 'questions_answers': questions_answers, 'choices': choices}

            print("ChoicesDict: ", choices)

            return render(request, 'upload_pdf.html', {'paragraph_questions': paragraph_questions})

        except Exception as e:
            return HttpResponse(f"Error extracting text from PDF: {e}")

    return render(request, 'upload_pdf.html')

