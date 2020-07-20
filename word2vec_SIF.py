from gensim.models import word2vec
import pandas as pd
import jieba.posseg
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from flask import Flask, render_template, request

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

def generateStopwords():
    stopwords = [line.strip() for line in open('cn_stopwords.txt', encoding='UTF-8').readlines()]
    return stopwords


def getCorpus(pathFrom):
    infoNum = 1199
    stopwords = generateStopwords()
    punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
    data = pd.read_excel(pathFrom)
    answerKeyword = data['KL_CONTENT_KEYWORD'].values[0:infoNum].astype(str)
    question = data['KL_CONTENT_TITLE'].values[0:infoNum].astype(str)
    answer = data['KL_CONTENT'].values[0:infoNum]
    sentences = []
    for i in range(infoNum):
        eachQuestion = []
        thisKeyword = jieba.posseg.cut(answerKeyword[i])
        thisQuestion = jieba.posseg.cut(question[i])
        for j in thisKeyword:
            if j.word not in stopwords and j.word not in punc and j.flag != 'm' and not j.word.isdigit():
            # if j.word not in punc and j.flag != 'm' and not j.word.isdigit():
                eachQuestion.append(j.word)
        for j in thisQuestion:
            if j.word not in stopwords and j.word not in punc and j.flag != 'm' and not j.word.isdigit():
            # if j.word not in punc and j.flag != 'm' and not j.word.isdigit():
                eachQuestion.append(j.word)
        sentences.append(eachQuestion)
    return sentences, question, answer


def getModel(sentences):
    if os.path.exists('word2vec'):
        model = word2vec.Word2Vec.load('word2vec')
    else:
        model = word2vec.Word2Vec(sentences, sg=0, size=dimension, window=5, min_count=1, negative=3, sample=0.001, workers=4)
        # model = word2vec.Word2Vec(sentences, sg=0, size=100, window=5, min_count=1, hs=1, workers=4)
        model.save('word2vec')
        # model.wv.save_word2vec_format('word2vec_txt', binary=False)
    return model


def getInput(inputSentence):
    stopwords = generateStopwords()
    punc = "，。、【 】:“”：；（）《》‘’{}？！⑦()、%^>℃：.”“^-——=&#@￥"
    sentence = jieba.posseg.cut(inputSentence)
    word = []
    for i in sentence:
        if i.word not in stopwords and i.word not in punc and i.flag != 'm' and not i.word.isdigit():
        # if i.word not in punc and i.flag != 'm' and not i.word.isdigit():
            word.append(i.word)
    return word, inputSentence


def getSentenceSIF(sentences, model):
    a = 0.001
    sentenceNum = len(sentences)
    sentenceVectors = np.zeros((sentenceNum, dimension))
    for i in range(sentenceNum):
        sentence = sentences[i]
        wordNum = len(sentence)
        count = {}
        sentenceVector = np.zeros((wordNum, dimension))
        for word in sentence:
            count[word] = count.get(word, 0) + 1
        probability = np.zeros(wordNum)
        for j in range(wordNum):
            probability[j] = count[sentence[j]] / wordNum
            sentenceVector[j] = model.wv[sentence[j]] * (a / (a + probability[j]))
            sentenceVectors[i] += sentenceVector[j]
        sentenceVectors[i] /= wordNum
    return sentenceVectors


def getInputSIF(inputSentence, model):
    a = 0.0001
    count = {}
    wordNum = len(inputSentence)
    inputVector = np.zeros((wordNum, dimension))
    probability = np.zeros(wordNum)
    SIFVector = np.zeros(dimension)
    for word in inputSentence:
        count[word] = count.get(word, 0) + 1
    for i in range(wordNum):
        probability[i] = count[inputSentence[i]] / wordNum
        inputVector[i] = model.wv[inputSentence[i]] * (a / (a + probability[i]))
        SIFVector += inputVector[i]
    SIFVector /= wordNum
    return SIFVector


def getPCA(sentencesVectors, SIFvector):
    allSentences = np.zeros((len(sentencesVectors) + 1, dimension))
    for i in range(len(sentencesVectors)):
        allSentences[i] = sentencesVectors[i]
    allSentences[-1] = SIFvector
    pca = PCA(n_components=dimension)
    pca.fit(allSentences)
    u = pca.components_[0]
    u = np.multiply(u, np.transpose(u))
    for i in range(len(allSentences)):
        common = np.multiply(u, allSentences[i])
        allSentences[i] = np.subtract(allSentences[i], common)
    totalSentencesVectors = []
    for i in range(len(sentencesVectors)):
        temp = []
        temp.append(allSentences[i])
        temp.append(allSentences[-1])
        totalSentencesVectors.append(temp)
    return totalSentencesVectors


def main(inputQuestion):
    output = {}
    output['inputQ'] = inputQuestion
    try:
        # inputSentence = input("please input a question:")
        inputSentence = inputQuestion
        inputInfo = getInput(inputSentence)
        inputToken = inputInfo[0]
        # inputSentence = inputInfo[1]
        if len(inputToken) == 0:
            raise ValueError

        inputSIF = getInputSIF(inputToken, model)
        allSIF = getPCA(sentenceSIF, inputSIF)
        typeNum = len(sentences)
        typeSimilarity = np.zeros(typeNum)
        for i in range(typeNum):
            typeSimilarity[i] = cosine_similarity(allSIF[i])[0][1]

        allIndex = np.arange(len(typeSimilarity))
        index = allIndex[typeSimilarity > threshhold]
        satisfiedNum = len(index)
        satisfiedSimilarity = np.zeros(satisfiedNum)
        for i in range(satisfiedNum):
            satisfiedSimilarity[i] = typeSimilarity[index[i]]

        similarity = {}
        if satisfiedNum >= 3:
            top_k = 3
            top_k_index = satisfiedSimilarity.argsort()[::-1][0:top_k]
            # print("input sentence:", inputSentence)
            # print()
            for i in range(top_k):
                # print((i + 1), "th match question with question ID", index[top_k_index[i]], ":", question[index[top_k_index[i]]])
                # print("similarity:", typeSimilarity[index[top_k_index[i]]])
                # print("answer:", answer[index[top_k_index[i]]])
                # print('--------------------------------')
                # similarity['input question'] = inputQuestion
                similarity.setdefault(i, {})['question'] = question[index[top_k_index[i]]]
                similarity.setdefault(i, {})['similarity'] = typeSimilarity[index[top_k_index[i]]]
                similarity.setdefault(i, {})['answer'] = answer[index[top_k_index[i]]]
        else:
            top_k_index = satisfiedSimilarity.argsort()[::-1]
            # print("input sentence:", inputSentence)
            # print()
            for i in range(satisfiedNum):
                # print((i + 1), "th match question with question ID", index[top_k_index[i]], ":", question[index[top_k_index[i]]])
                # print("question:", question[index[top_k_index[i]]])
                # print("answer:", answer[index[top_k_index[i]]])
                # print('--------------------------------')
                # similarity['input question'] = inputQuestion
                similarity.setdefault(i, {})['question'] = question[index[top_k_index[i]]]
                similarity.setdefault(i, {})['similarity'] = typeSimilarity[index[top_k_index[i]]]
                similarity.setdefault(i, {})['answer'] = answer[index[top_k_index[i]]]
        output['dict'] = similarity
    except KeyError:
        prompt = "sorry, can not find any suitable answer for this sentence"
        output['prompt'] = prompt
        output['check'] = 1
        # print("sorry, can not find any suitable answer for this sentence")
    except ValueError:
        prompt = "sorry, this question has wrong format"
        output['prompt'] = prompt
        output['check'] = 2
        # print("sorry, this question has wrong format")
    else:
        prompt = "the most similar questions and their answers have been given"
        output['prompt'] = prompt
        output['check'] = 0
        # print("the most 3 similar questions and their ansewrs have been given")
    # output.append(inputQuestion)
    return output


@app.route('/index', methods=['GET'])
def index():
    # QA_question = request.form.get('question')
    # QA_output = main(QA_question)
    # print(QA_output)
    # return QA_output
    # print(QA_output[0])
    # return str(len(QA_output[0]))
    # return QA_output

    # return render_template('test.html', inputQuestion=QA_output[0]['input question'], firstMatchQuestion=QA_output[0]['1 th match']['question'],\
    #                        firstMatchSimilarity=QA_output[0]['1 th match']['similarity'],\
    #                        firstMatchAnswer=QA_output[0]['1 th match']['answer'], \
    #                        secondMatchQuestion=QA_output[0]['2 th match']['question'],\
    #                        secondMatchSimilarity=QA_output[0]['2 th match']['similarity'],\
    #                        secondMatchAnswer=QA_output[0]['2 th match']['answer'],\
    #                        thirdMatchQuestion=QA_output[0]['3 th match']['question'],\
    #                        thirdMatchSimilarity=QA_output[0]['3 th match']['similarity'],\
    #                        thirdMatchAnswer=QA_output[0]['3 th match']['answer'],\
    #                        prompt=QA_output[1]
    #                        )
    # print(QA_question, i)
    # i+=1
    return render_template('test.html')
    # return 'finished'


@app.route('/test', methods=['POST'])
def QA():
    QA_question = request.form.get('question')
    QA_output = main(QA_question)
    print(QA_output)
    return QA_output


# jieba.load_userdict('new_dict.txt')
dimension = 500
info = getCorpus('问答类信息20200619.xlsx')
sentences = info[0]
question = info[1]
answer = info[2]
model = getModel(sentences)
sentenceSIF = getSentenceSIF(sentences, model)
threshhold = 0.4


if __name__ == '__main__':
    # main()
    app.run(host='127.0.0.1', port=5000, debug=True)