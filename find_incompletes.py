from utils import write_incomplete_data


def main():
    write_incomplete_data('data/json/doi_articles.json',
                          outfile='incomplete_from_doi.csv')
    write_incomplete_data('data/json/salvaged_articles.json',
                          outfile='incomplete_from_salvaged.csv')


if __name__ == '__main__':
    main()
