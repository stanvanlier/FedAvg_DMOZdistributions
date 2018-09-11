import pkg_resources

ROOT_DIR = pkg_resources.resource_filename("src", "..")


def read_tsv(file):
    """
    Read `.tsv` file to pandas DataFrame

    :param file: path to `.tsv` file
    :return: pandas Dataframe
    """
    import pandas as pd

    return pd.read_table(file, quoting=3)


def parse_cat_level(s, n):
    start_idx = s.find("/")
    for _ in range(n):
        if start_idx == len(s):
            return None
        start_idx = s.find("/", start_idx + 1)
        if start_idx == -1:
            start_idx = len(s)
    return s[:start_idx]


def get_selected_content_df():
    import os

    content_file = os.path.join(ROOT_DIR, "data/interim/content.tsv")
    if not os.path.exists(content_file):
        from src.data import get_data

        print("TSV not found, trying to create it.")
        get_data.make_content_tsv(which="main")

    df = read_tsv(content_file)[["topic", "title", "description"]]
    df["label"] = df.topic.apply(parse_cat_level, args=[1])
    df = df[df.label.map(
        lambda x: x not in ['Top/World', 'Top/Regional', 'Top/News']
    )]
    return df


def get_selected_structure_df():
    import os

    structure_file = os.path.join(ROOT_DIR, "data/interim/structure.tsv")
    if not os.path.exists(structure_file):
        from src.data import get_data

        print("TSV not found, trying to create it.")
        get_data.make_structure_tsv(which="main")

    df = read_tsv(structure_file)
    df = df[df.tag.map(
        lambda x: x.startswith(("narrow", "symbolic", "related"))
    )]
    df.resource = df.resource.map(
        lambda x: x[x.find(":") + 1:]
    )
    return df


def pick_topics(df, structure_df, min_pages=200):
    import pandas as pd

    picked_df = df[["title", "topic"]].groupby("topic").count()
    picked_df.columns = ["pages"]

    topics_having_relations = pd.Index(structure_df.topic.append(structure_df.resource).unique())
    print('start', len(picked_df))
    picked_df['level'] = picked_df.index.map(lambda x: x.count("/"))
    print("for: ", end="")
    for i in reversed(range(2, picked_df.level.max() + 1)):
        picked_df["has_relations"] = picked_df.index.map(topics_having_relations.contains)
        picked_df['parent'] = picked_df.index.map(lambda x: x[:x.rfind("/")])
        picked_df['topic_group'] = picked_df.apply(
            lambda x: x.parent if x.level == i and (not x.has_relations or x.pages < min_pages) else x.name,
            axis=1)
        picked_df = picked_df[["pages", "topic_group"]].groupby("topic_group").sum()
        picked_df.index.name = "topic"
        picked_df['level'] = picked_df.index.map(lambda x: x.count("/"))
        print(i, len(picked_df), end="|")

    del topics_having_relations

    relations_of_picked = structure_df[structure_df.topic.map(picked_df.index.contains)
                                       & structure_df.resource.map(picked_df.index.contains)]
    picked_having_relations = pd.Index(relations_of_picked.topic.append(relations_of_picked.resource).unique())
    picked_df["has_inner_relations"] = picked_df.index.map(picked_having_relations.contains)
    del picked_having_relations

    while not picked_df.has_inner_relations.all():
        print("\nwhile: ", end="")
        for i in reversed(range(2, picked_df.level.max() + 1)):
            picked_df['parent'] = picked_df.index.map(lambda x: x[:x.rfind("/")])
            picked_df['topic_group'] = picked_df.apply(
                lambda x: x.parent if x.level == i and x.has_inner_relations == False else x.name,
                axis=1)
            picked_df = picked_df[["pages", "topic_group"]].groupby("topic_group").sum()
            picked_df.index.name = "topic"

            relations_of_picked = structure_df[structure_df.topic.map(picked_df.index.contains)
                                               & structure_df.resource.map(picked_df.index.contains)]
            topics_in_SpanningTree = set()
            prev_added_topics = set(["Top/Arts"])
            while len(prev_added_topics) > 0:
                added_topics = set()
                for x in prev_added_topics:
                    added_topics.update(
                        relations_of_picked[relations_of_picked.topic == x].resource.values
                    )
                    added_topics.update(
                        relations_of_picked[relations_of_picked.resource == x].topic.values
                    )
                prev_added_topics = added_topics.difference(topics_in_SpanningTree)
                topics_in_SpanningTree.update(prev_added_topics)
            picked_df["has_inner_relations"] = picked_df.index.map(pd.Index(topics_in_SpanningTree).contains)
            picked_df['level'] = picked_df.index.map(lambda x: x.count("/"))
            print(i, len(picked_df), end="|")
            if picked_df.has_inner_relations.all():
                print("breaked. all picked topics are in spanning tree")
                break

    return picked_df, relations_of_picked


def main():
    import pandas as pd
    import numpy as np
    import os

    # processed_path = "../../data/processed"
    # interim_path = os.path.join(ROOT_DIR, "data/interim")
    processed_path = os.path.join(ROOT_DIR, "data/processed")
    os.makedirs(os.path.join(ROOT_DIR, processed_path), exist_ok=True)

    all_df = get_selected_content_df()
    labels_df = pd.DataFrame({"y": range(12)}, index=all_df.label.unique())
    all_df["y"] = all_df.label.map(labels_df.y)

    labels_df.to_pickle(os.path.join(processed_path, "labels_df.pkl"))

    relations_df = get_selected_structure_df()

    picked_df, relations_df = pick_topics(all_df, relations_df)


    relations_df.to_pickle(os.path.join(processed_path, "relations_df.pkl"))

    picked_df["node_id"] = range(len(picked_df))
    picked_df.to_pickle(os.path.join(processed_path, "picked_df.pkl"))

    # find all relations for each node
    nodes_df = picked_df.reset_index().set_index("node_id")[["topic"]]
    nodes_df["relations"] = nodes_df.topic.map(
        lambda x: relations_df[(relations_df.topic == x)].resource.append(
            relations_df[(relations_df.resource == x)].topic
        ).map(lambda y: picked_df.loc[y, 'node_id']).unique()
    )

    P = 2  # P = ratio of the probability for a page to stay on the same node or to go to one other node
    nodes_df['swap_to'] = nodes_df.apply(lambda x: list(x.relations) + [x.name] * P, axis=1)

    nodes_df.to_pickle(os.path.join(processed_path, "nodes_df.pkl"))

    def topic2picked(topic):
        while True:
            if picked_df.index.contains(topic):
                return topic
            elif topic == "":
                return "NOT FOUND"
            topic = topic[:topic.rfind("/")]

    all_df["node_topic"] = all_df.topic.map(topic2picked)

    all_df.to_pickle(os.path.join(processed_path, "all_df.pkl"))

    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, random_state=42, test_size=0.3)
    train_index, test_index = next(sss.split(np.zeros(len(all_df)), all_df.node_topic.values))
    train_df = all_df.iloc[train_index]
    test_df = all_df.iloc[test_index]

    train_df.to_pickle(os.path.join(processed_path, "train_df.pkl"))
    test_df.to_pickle(os.path.join(processed_path, "test_df.pkl"))

    def make_train_distributions(until):
        dist_df = pd.DataFrame(index=train_df.index)
        np.random.seed(42)
        for d in range(0, until + 1):
            if d == 0:
                dist_df["d0"] = train_df.node_topic.map(lambda x: picked_df.loc[x, "node_id"])
            else:
                dist_df["d{}".format(d)] = dist_df["d{}".format(d - 1)].map(nodes_df.swap_to).apply(np.random.choice)
            print(d, end=" ")
            if d % 40 == 0:
                print()
        np.random.seed(510)
        dist_df["d-1"] = np.random.randint(0, len(picked_df), len(train_df))
        print("-1")
        return dist_df

    make_train_distributions(100).to_pickle(os.path.join(processed_path, "train_distributions_df.pkl"))
    # train_df.to_pickle(os.path.join(processed_path, "train_df.pkl"))


if __name__ == "__main__":
    main()
