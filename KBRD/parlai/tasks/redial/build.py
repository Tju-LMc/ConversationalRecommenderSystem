import csv
import json
import os
import pickle as pkl
import random
import re
from collections import defaultdict

import parlai.core.build_data as build_data


def _split_data(redial_path):
    # Copied from https://github.com/RaymondLi0/conversational-recommendations/blob/master/scripts/split-redial.py
    data = []
    for line in open(os.path.join(redial_path, "train_data.jsonl")):
        data.append(json.loads(line))
    random.shuffle(data)
    n_data = len(data)
    split_data = [data[: int(0.9 * n_data)], data[int(0.9 * n_data) :]]

    with open(os.path.join(redial_path, "train_data.jsonl"), "w") as outfile:
        for example in split_data[0]:
            json.dump(example, outfile)
            outfile.write("\n")
    with open(os.path.join(redial_path, "valid_data.jsonl"), "w") as outfile:
        for example in split_data[1]:
            json.dump(example, outfile)
            outfile.write("\n")


def _entity2movie(entity, abstract=""):
    # strip url
    x = entity[::-1].find("/")  # 电影名称倒序起始索引
    movie = entity[-x:-1]       # 电影名称
    movie = movie.replace("_", " ") 

    # extract year
    pattern = re.compile(r"\d{4}")
    match = re.findall(pattern, movie)
    year = match[0] if match else None  # 电影名称中的电影年份
    # if not find in entity title, find in abstract
    if year is None:                    # 电影名称中没有则去摘要中找
        pattern = re.compile(r"\d{4}")
        match = re.findall(pattern, abstract)
        if match and 1900 < int(match[0]) and int(match[0]) < 2020:
            year = match[0]

    # recognize (20xx film) or (film) to help disambiguation
    pattern = re.compile(r"\(.*film.*\)")
    match = re.findall(pattern, movie)
    definitely_is_a_film = match != [] # 电影名称中有没有标明是电影

    # remove parentheses              # 删除括号
    while True:
        pattern = re.compile(r"(.+)( \(.*\))")
        match = re.search(pattern, movie)
        if match:
            movie = match.group(1)
        else:
            break
    movie = movie.strip()    # 拿到准确的电影名称

    return movie, year, definitely_is_a_film  # 返回电影名称，年份，是不是真的是电影


DBPEDIA_ABSTRACT_PATH = "/home/tjucic/zjf/KBRD/dbpedia/short_abstracts_en.ttl"
DBPEDIA_PATH = "/home/tjucic/zjf/KBRD/dbpedia/mappingbased_objects_en.ttl"


def _build_dbpedia(dbpedia_path):
    movie2entity = {}
    movie2years = defaultdict(set)
    with open(dbpedia_path) as f:
        for line in f.readlines():    # line格式："entity xxx abstract"
            if line.startswith("#"):
                continue
            entity, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            _, line = line[: line.index(" ")], line[line.index(" ") + 1 :]
            abstract = line[:-4]
            movie, year, definitely_is_a_film = _entity2movie(entity, abstract)
            if (movie, year) not in movie2entity or definitely_is_a_film:
                movie2years[movie].add(year)
                movie2entity[(movie, year)] = entity
    return {"movie2years": movie2years, "movie2entity": movie2entity}


def _load_kg(path):
    kg = defaultdict(list)
    with open(path) as f:
        for line in f.readlines():
            tuples = line.split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                # TODO: include property/publisher and subject/year, etc
                if "ontology" in r:
                    kg[h].append((r, t))
    return kg


def _extract_subkg(kg, seed_set, n_hop):
    subkg = defaultdict(list)
    subkg_hrt = set()

    ripple_set = []
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = seed_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                h, r, t = entity, tail_and_relation[0], tail_and_relation[1]
                if (h, r, t) not in subkg_hrt:
                    subkg[h].append((r, t))
                    subkg_hrt.add((h, r, t))
                memories_h.append(h)
                memories_r.append(r)
                memories_t.append(t)

        ripple_set.append((memories_h, memories_r, memories_t))

    return subkg


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "redial")
    # define version if any
    version = None

    # check if data had been previously built
    if not build_data.built(dpath, version_string=version):
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # # download the data.
        # fname = "redial_dataset.zip"
        # url = "https://github.com/ReDialData/website/raw/data/" + fname  # dataset URL
        # build_data.download(url, dpath, fname)

        # # uncompress it
        # build_data.untar(dpath, fname)

        _split_data(dpath)

        dbpedia = _build_dbpedia(DBPEDIA_ABSTRACT_PATH)
        movie2entity = dbpedia["movie2entity"]
        movie2years = dbpedia["movie2years"]

        # Match REDIAL movies to dbpedia entities
        movies_with_mentions_path = os.path.join(dpath, "movies_with_mentions.csv")
        with open(movies_with_mentions_path, "r") as f:
            reader = csv.reader(f)
            id2movie = {int(row[0]): row[1] for row in reader if row[0] != "movieId"}
        id2entity = {}
        for movie_id in id2movie:
            movie = id2movie[movie_id]
            pattern = re.compile(r"(.+)\((\d+)\)")
            match = re.search(pattern, movie) # 找到匹配的movie名称和movie年份
            if match is not None:
                name, year = match.group(1).strip(), match.group(2)
            else:
                name, year = movie.strip(), None
            if year is not None:   # 如果数据集中的电影有年份，获取 movie_id 对应的 dbpedia中的 entity_id
                if (name, year) in movie2entity:
                    id2entity[movie_id] = movie2entity[(name, year)]
                else:
                    if len(movie2years) == 1:
                        id2entity[movie_id] = movie2entity[(name, movie2years[name][0])]
                    else:
                        id2entity[movie_id] = None

            else:
                id2entity[movie_id] = (
                    movie2entity[(name, year)] if (name, year) in movie2entity else None
                )
        # HACK: make sure movies are matched to different entities
        matched_entities = set()  # 获取能在知识图谱中匹配的电影实体id
        for movie_id in id2entity:
            if id2entity[movie_id] is not None:
                if id2entity[movie_id] not in matched_entities:
                    matched_entities.add(id2entity[movie_id])
                else:
                    id2entity[movie_id] = None

        # Extract sub-kg related to movies
        kg = _load_kg(DBPEDIA_PATH)
        subkg = _extract_subkg(
            kg,
            [
                id2entity[k]
                for k in id2entity
                if id2entity[k] is not None and kg[id2entity[k]] != []
            ],
            2,  # 从entity_id出发，经过两跳构成的子图
        )
        for movie_id in id2entity:
            if id2entity[movie_id] is not None:
                subkg[id2entity[movie_id]].append(('self_loop', id2entity[movie_id]))
            else:
                subkg[movie_id].append(('self_loop', movie_id))
        entities = set([k for k in subkg]) | set([x[1] for k in subkg for x in subkg[k]]) # 获取子图中所有的实体
        entity2entityId = dict([(k, i) for i, k in enumerate(entities)]) # 构建entity的id
        relations = set([x[0] for k in subkg for x in subkg[k]])  # 获取子图中所有的relation
        relation2relationId = dict([(k, i) for i, k in enumerate(relations)]) #构建relation的id
        subkg_idx = defaultdict(list) # 子图中实体映射到id
        for h in subkg:
            for r, t in subkg[h]:
                subkg_idx[entity2entityId[h]].append((relation2relationId[r], entity2entityId[t]))
        movie_ids = [] # 电影实体映射到对应的id
        for k in id2entity:
            movie_ids.append(entity2entityId[id2entity[k]] if id2entity[k] is not None else entity2entityId[k])

        pkl.dump(id2entity, open(os.path.join(dpath, "id2entity.pkl"), "wb"))
        pkl.dump(dbpedia, open(os.path.join(dpath, "dbpedia.pkl"), "wb"))
        pkl.dump(subkg_idx, open(os.path.join(dpath, "subkg.pkl"), "wb"))
        pkl.dump(entity2entityId, open(os.path.join(dpath, "entity2entityId.pkl"), "wb"))
        pkl.dump(relation2relationId, open(os.path.join(dpath, "relation2relationId.pkl"), "wb"))
        pkl.dump(movie_ids, open(os.path.join(dpath, "movie_ids.pkl"), "wb"))

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
