from nltk.corpus import dependency_treebank
def main():
    S = dependency_treebank.parsed_sents()

    for tree in dependency_treebank.parsed_sents():
        V = tree.nodes
        for node in V:
            print(node['word'])

if __name__ == '__main__':
    main()