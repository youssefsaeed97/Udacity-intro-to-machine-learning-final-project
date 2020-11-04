import _pickle as cPickle

def pkledit(original):

    original = original + ".pkl"
    destination = original + "_word_data_unix.pkl"
    content = ''
    outsize = 0
    with open(original, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize = outsize + len(line) + 1
            output.write(line + str.encode('\n'))

    words_file_handler = open(destination, "rb")
    word_data = cPickle.load(words_file_handler)
    words_file_handler.close()

    return destination
