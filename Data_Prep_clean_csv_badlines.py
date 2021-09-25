for i in range(1,5):
    j = str(i)
    with open("data/pubmed_articles_cancer_0" + j + ".csv", 'r') as file:
        lines = file.readlines()

    # delete matching content
    content = ",,,,,"
    with open("data/pubmed_articles_cancer_0" + j + "_smaller.csv", 'w') as file:
        for line in lines:
            # readlines() includes a newline character
            if line.strip("\n") != content:
                file.write(line)

