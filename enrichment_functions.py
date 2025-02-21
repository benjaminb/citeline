def add_abstract(record, sentences):
    abstract = record['abstract']
    return [abstract + '\n' + sentence for sentence in sentences]


def add_title(record, sentences):
    title = record['title']
    return [title + '\n' + sentence for sentence in sentences]


def add_title_and_abstract(record, sentences):
    title = record['title']
    abstract = record['abstract']
    return [title + '\n' + abstract + '\n' + sentence for sentence in sentences]


def no_augmentation(record, sentences):
    return sentences
