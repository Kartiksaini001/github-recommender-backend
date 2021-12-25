from flask import Flask
from flask_cors import CORS
import re
import base64
import json
import requests
# import os
import pickle as pkl
import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
# from scipy.sparse import vstack, csr_matrix, save_npz, load_npz
from scipy.sparse import csr_matrix, load_npz
import datetime
import time
import operator

app = Flask(__name__)
CORS(app)

github_tokens = ["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",
                 "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"]
ptr = 0
github_token = github_tokens[ptr]
MOD = 10007
MOD_space = []
for i in range(MOD):
    MOD_space.append(str(i))
MOD_space = ' '.join(MOD_space)

# stop_procedure = False
# stop_index = bg - 1

source_extentions = ['py', 'ipynb', 'cpp', 'c', 'cfg', 'js',
                     'json', 'vue', "xml", "java", "sh", "php", "rb", "ts"]
text = []
source = []


def github_read_file(username, repository_name, file_path):
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"

    url = f'https://api.github.com/repos/{username}/{repository_name}/contents/{file_path}'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
    except:
        return ""
    file_content = data['content']
    file_content_encoding = data.get('encoding')
    if file_content_encoding == 'base64':
        file_content = base64.b64decode(file_content).decode()
    return file_content


def get_files(username, repository_name, file_path):
    file_content = github_read_file(username, repository_name, file_path)
    return file_content


def get_contents(username, repository_name):
    headers = {}
    global github_token
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    url = f'https://api.github.com/repos/{username}/{repository_name}/contents'
    r = requests.get(url)
    try:
        r.raise_for_status()
        data = r.json()
        return data
    except:
        return []


def recur(user, repo, r):
    headers = {}
    global github_token
    global text, source
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    for i in r.json():
        if i["name"].startswith('.'):
            continue
        if i['type'] == 'file':
            if i['name'].split('.')[-1] == 'md' or i['name'].split('.')[-1] == 'txt':
                text.append(i)
            elif i['name'].split('.')[-1] in source_extentions:
                source.append(i)
        elif i['type'] == 'dir':
            url = f'https://api.github.com/repos/{user}/{repo}/contents/' + \
                i['path']
            try:
                r = requests.get(url, headers=headers)
                r.raise_for_status()
                recur(user, repo, r)
            except:
                pass


def get_filenames(user, repo):
    headers = {}
    global github_token
    global source, text, stop_procedure
    source = []
    text = []
    if github_token:
        headers['Authorization'] = f"token {github_token}"
    url = f'https://api.github.com/repos/{user}/{repo}/contents'
    try:
        r = requests.get(url)
        r.raise_for_status()
        recur(user, repo, r)
    except:
        try:
            print("error @ get_filenames", user, repo)
            if r.json()['message'] != 'This repository is empty.':
                #print("Empty nahi hai. hum nahi denge kya kar loge")
                stop_procedure = True
        except:
            print("Dual Error")
            print(r, r.json())


def extract_words(user, repo, text, k):
    puntuation = '1234567890-=!@#$%^&*()+[]{};:"\'|\\<>~/?`<>.,'
    out = ""
    ps = PorterStemmer()
    for i in text:
        k -= 1
        if k == 0:
            return out
        try:
            data = get_files(user, repo, i['path'])
            # print(data)
            for i in puntuation:
                data = data.replace(i, " ")
            data = data.replace("\n", ' ')
            data = re.sub(r'\b\w{1,2}\b', '', data)
            data += ' '
            data = re.sub("\s\s+", " ", data)
            if len(data) > 0 and data[0] == ' ':
                data = data[1:]
            data = data.lower()
            ps.stem(data)
            data = ' '.join([str(hash(i) % MOD) for i in data.split(' ')])
            out += data + ' '
        except:
            print("error reading", i["name"])
            pass
    return out


# find collaborators route
@app.route("/repos/<username>/<repo>", methods=['GET', 'POST'])
def find_collaborators(username, repo):

    get_filenames(username, repo)
    text_hashed = extract_words(username, repo, text, 3)
    source_hashed = extract_words(username, repo, source, 7)

    vocab = {}
    for i in range(MOD):
        vocab[str(i)] = i

    def get_mapper(voc):
        def mapper(key):
            if key not in voc:
                key = '0'
            return voc[key]
        return mapper

    mapper = get_mapper(vocab)
    i = text_hashed
    res = map(lambda x: mapper(x), i.split(' '))
    temp = np.bincount(list(res), minlength=MOD)
    if len(i.split(' ')) > 0:
        temp = temp / np.array([len(i.split(' '))], dtype=np.float64)
    Repo_text_point = csr_matrix(temp, dtype=np.float64)

    i = source_hashed
    res = map(lambda x: mapper(x), i.split(' '))
    temp = np.bincount(list(res), minlength=MOD)
    if len(i.split(' ')) > 0:
        temp = temp / np.array([len(i.split(' '))], dtype=np.float64)
    Repo_source_point = csr_matrix(temp, dtype=np.float64)

    idf_text = np.load('Lib/idf_text.npy')
    idf_source = np.load('Lib/idf_source.npy')

    Repo_source_point_2d = Repo_source_point.toarray()  # Convert 1D output to 2D
    tf_idf_source = idf_source * Repo_source_point_2d
    Repo_text_point_2d = Repo_text_point.toarray()  # Convert 1D output to 2D
    tf_idf_text = idf_text * Repo_text_point_2d

    # tfidf_source = csr_matrix(load_npz("Lib/tfidf_source.npz"))
    # tfidf_text = csr_matrix(load_npz("Lib/tfidf_text.npz"))
    UB_matrix = load_npz("Lib/UB_matrix.npz")

    ind_usr = np.where(np.load("Lib/valid_users.npy"))[0]
    ind_repo = np.where(np.load("Lib/valid_repos.npy"))[0]

    UB_matrix = UB_matrix[ind_usr[:, np.newaxis], ind_repo]

    with open("Lib/users.txt", 'r') as f:
        usr = [x.strip() for x in f]
    usr = np.array(usr)[ind_usr]

    with open("Lib/repos.txt", 'r') as f:
        txt = [x.strip() for x in f]
    txt = np.array(txt)[ind_repo]

    model_text = pkl.load(open("Lib/mtext", "rb"))
    model_source = pkl.load(open("Lib/msource", "rb"))

    out1 = model_text.kneighbors(tf_idf_text)
    out2 = model_source.kneighbors(tf_idf_source)

    recommend1 = sorted([(x[0], y) for x, y in zip(np.dot(
        UB_matrix[:, out1[1][0]].toarray(), (1 - out1[0]).T), usr)], reverse=True)
    recommend2 = sorted([(x[0], y) for x, y in zip(np.dot(
        UB_matrix[:, out2[1][0]].toarray(), (1 - out2[0]).T), usr)], reverse=True)

    temp_list = []

    for i in range(0, 10):
        temp_list.append(recommend2[i][1])

    for i in range(0, 10):
        temp_list.append(recommend1[i][1])

    final_list = []
    for i in temp_list:
        if i not in final_list:
            final_list.append(i)

    return_object = {}
    for username in final_list:
        req = requests.get("http://api.github.com/users/" + username)
        json = req.json()
        return_object[username] = [
            json['name'], json['followers'], json['bio']]

    return {"collaborators": return_object}


# find repos route
@app.route("/users/<username>", methods=['GET', 'POST'])
def find_repos(username):

    req = requests.get("http://api.github.com/users/"+username+"/repos")
    json = req.json()

    user_project = {}
    error_object = "Retrieval Failed."
    for i in json:
        if 'message' in json:
            print(error_object)
            break
        user_project[i['name']] = [i['stargazers_count'], i['watchers_count'],
                                   i['forks_count'], i['created_at'], i['updated_at'], i['pushed_at']]

    alpha = 30000000
    beta = 10000000
    gamma = 40000000
    project_priority = {}
    for i in user_project.keys():
        project_priority[i] = alpha * user_project[i][0] + \
            beta * user_project[i][1] + gamma * user_project[i][2]
        obj = datetime.datetime.strptime(
            user_project[i][3], "%Y-%m-%dT%H:%M:%SZ")
        t1 = time.mktime(obj.timetuple())
        obj = datetime.datetime.strptime(
            user_project[i][4], "%Y-%m-%dT%H:%M:%SZ")
        t2 = time.mktime(obj.timetuple())
        obj = datetime.datetime.strptime(
            user_project[i][5], "%Y-%m-%dT%H:%M:%SZ")
        t3 = time.mktime(obj.timetuple())
        project_priority[i] += max(t1, t2, t3)

    keyMax = max(project_priority.items(), key=operator.itemgetter(1))[0]

    max2 = 0
    keyMax2 = ""
    for v in project_priority.keys():
        if(project_priority[v] > max2 and project_priority[v] < project_priority[keyMax]):
            max2 = project_priority[v]
            keyMax2 = v

    repo_list = [keyMax, keyMax2]
    repo_list = [username + '/' + rep for rep in repo_list]
    repo_list.remove(repo_list[1])

    model_text = pkl.load(open("Lib/mtext", "rb"))
    model_source = pkl.load(open("Lib/msource", "rb"))

    out1 = []
    out2 = []

    def recommender():
        vocab = {}
        for i in range(MOD):
            vocab[str(i)] = i

        def get_mapper(voc):
            def mapper(key):
                if key not in voc:
                    key = '0'
                return voc[key]
            return mapper

        mapper = get_mapper(vocab)
        i = text_hashed
        res = map(lambda x: mapper(x), i.split(' '))
        temp = np.bincount(list(res), minlength=MOD)
        if len(i.split(' ')) > 0:
            temp = temp / np.array([len(i.split(' '))], dtype=np.float64)
        Repo_text_point = csr_matrix(temp, dtype=np.float64)

        i = source_hashed
        res = map(lambda x: mapper(x), i.split(' '))
        temp = np.bincount(list(res), minlength=MOD)
        if len(i.split(' ')) > 0:
            temp = temp / np.array([len(i.split(' '))], dtype=np.float64)
        Repo_source_point = csr_matrix(temp, dtype=np.float64)

        idf_text = np.load('Lib/idf_text.npy')
        idf_source = np.load('Lib/idf_source.npy')

        Repo_source_point_2d = Repo_source_point.toarray()
        tf_idf_source = idf_source * Repo_source_point_2d

        Repo_text_point_2d = Repo_text_point.toarray()
        tf_idf_text = idf_text * Repo_text_point_2d

        # tfidf_source = csr_matrix(load_npz("Lib/tfidf_source.npz"))
        # tfidf_text = csr_matrix(load_npz("Lib/tfidf_text.npz"))
        UB_matrix = load_npz("Lib/UB_matrix.npz")

        ind_usr = np.where(np.load("Lib/valid_users.npy"))[0]
        ind_repo = np.where(np.load("Lib/valid_repos.npy"))[0]

        UB_matrix = UB_matrix[ind_usr[:, np.newaxis], ind_repo]

        with open("Lib/users.txt", 'r') as f:
            usr = [x.strip() for x in f]
        usr = np.array(usr)[ind_usr]

        with open("Lib/repos.txt", 'r') as f:
            txt = [x.strip() for x in f]
        txt = np.array(txt)[ind_repo]

        out1.append(model_text.kneighbors(tf_idf_text))
        out2.append(model_source.kneighbors(tf_idf_source))

    repo_indices = []
    for repository in repo_list:
        username = repository.split('/')[0]
        repo = repository.split('/')[1]
        get_filenames(username, repo)
        text_hashed = extract_words(username, repo, text, 2)
        source_hashed = extract_words(username, repo, source, 3)

        recommender()

        i = 0
        temp = out2[i][1][0][0:10]
        for idx in temp:
            repo_indices.append(idx)
        temp = out1[i][1][0][0:10]
        for idx in temp:
            repo_indices.append(idx)

    file = open("Lib/repos.txt")
    return_object = []
    for pos, l_num in enumerate(file):
        if pos in repo_indices:
            l_num = l_num.strip()
            owner = l_num.split('/')[0]
            repo = l_num.split('/')[1]
            return_object.append([repo, owner])
    file.close()

    return {"repos": return_object}


if __name__ == '__main__':
    app.run()
