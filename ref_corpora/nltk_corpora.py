import nltk
import teanga
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

# Download the NLTK data
#nltk.download('all')

detokenizer = TreebankWordDetokenizer()
detokenizer.ENDING_QUOTES = [
        (re.compile(r"([^' ])\s('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) "), r"\1\2 "),
        (re.compile(r"([^' ])\s('[sS]|'[mM]|'[dD]|') "), r"\1\2 "),
        (re.compile(r"([^'\s])\s(\'\')"), r"\1\2"),
        (
            re.compile(r"(\'\')\s([.,:)\]>};%])"),
            r"\1\2",
        ),  # Quotes followed by no-left-padded punctuations.
        (re.compile(r"''"), '"'),
    ]

def find_spans(text, tokens, offset=0):
    spans = []
    start = 0
    for token in tokens:
        if token == '``':
            token = "\""
        elif token == "''":
            token = "\""
        elif "''" in token:
            token = token.replace("''", "\"")
        start = text.find(token, start)
        if start == -1:
            raise ValueError("Token not found in text: " + token + 
                             " in " + text[start:])
        end = start + len(token)
        spans.append((start + offset, end + offset))
        start = end
    return spans

def convert_plain_text(corpus_name):
    corpus = teanga.Corpus()
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    for fileid in nltk_corpus.fileids():
        text = nltk_corpus.raw(fileid)
        doc = corpus.add_doc(text)
        doc.fileid = [fileid]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

def convert_tagged_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("tags", layer_type="seq", base="words",
                          data=list(set(tag for (word, tag) in 
                                        nltk_corpus.tagged_words())))
    sents = False
    try:
        nltk_corpus.sents()
        corpus.add_layer_meta("sentences", layer_type="div", base="words")
        sentences = True
    except Exception:
        pass

    paragraphs = False
    try:
        nltk_corpus.paras()
        corpus.add_layer_meta("paragraphs", layer_type="div", base="sentences")
        paragraphs = True
    except Exception:
        pass
    for fileid in nltk_corpus.fileids():
        if paragraphs:
            text = "\n\n".join(" ".join(
                detokenizer.detokenize(sent) 
                for sent in para) 
                for para in nltk_corpus.paras(fileid))
            doc = corpus.add_doc(text)
            offset = 0
            sents = []
            paras = []
            word_idxs = []
            for para in nltk_corpus.paras(fileid):
                paras.append(len(sents))
                for sent in para:
                    sents.append(len(word_idxs))
                    word_idxs.extend(find_spans(text[offset:], sent, offset))
                    offset += len(detokenizer.detokenize(sent))
                    offset += 1
                offset += 2
            doc.words = word_idxs
            doc.sentences = sents
            doc.paragraphs = paras
        elif sentences:
            text = " ".join(detokenizer.detokenize(sent)
                            for sent in nltk_corpus.sents(fileid))
            doc = corpus.add_doc(text)
            offset = 0
            sents = []
            word_idxs = []
            for sent in nltk_corpus.sents(fileid):
                sents.append(len(word_idxs))
                word_idxs.extend(find_spans(text[offset:], sent, offset))
                offset += len(detokenizer.detokenize(sent))
                offset += 1
            doc.words = word_idxs
            doc.sentences = sents
        else:
            words = nltk_corpus.words(fileid)
            text = detokenizer.detokenize(words)
            doc = corpus.add_doc(text)
            doc.fileid = [fileid]
            doc.words = find_spans(text, words)

        doc.tags = [tag for (word, tag) in nltk_corpus.tagged_words(fileid)]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

#def convert_dep_corpus(corpus_name):
#    corpus = teanga.Corpus()
#    nltk_corpus = eval("nltk.corpus." + corpus_name)
#    corpus.add_layer_meta("text")
#    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
#    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
#    corpus.add_layer_meta("words", layer_type="span", base="text")
#    corpus.add_layer_meta("tags", layer_type="seq", base="words",
#                          data=list(set(tag for (word, tag) in 
#                                        nltk_corpus.tagged_words())))
#    rel_types = set(
#            dep["rel"] for tree in nltk_corpus.parsed_sents()
#            for dep in tree.nodes.values()
#            if dep["rel"])
#    if rel_types:
#        corpus.add_layer_meta("dep", layer_type="seq", base="words",
#                              data="link", link_types=rel_types)
#    else:
#        corpus.add_layer_meta("dep", layer_type="seq", base="words",
#                              data="link")
#    sents = False
#    try:
#        nltk_corpus.sents()
#        corpus.add_layer_meta("sentences", layer_type="div", base="words")
#        sentences = True
#    except Exception:
#        pass
#
#    paragraphs = False
#    try:
#        nltk_corpus.paras()
#        corpus.add_layer_meta("paragraphs", layer_type="div", base="sentences")
#        paragraphs = True
#    except Exception:
#        pass
#    for fileid in nltk_corpus.fileids():
#        if paragraphs:
#            text = "\n\n".join(" ".join(
#                detokenizer.detokenize(sent) 
#                for sent in para) 
#                for para in nltk_corpus.paras(fileid))
#            doc = corpus.add_doc(text)
#            offset = 0
#            sents = []
#            paras = []
#            word_idxs = []
#            deps = []
#            for para in nltk_corpus.parsed_paras(fileid):
#                paras.append(len(sents))
#                for sent in para:
#                    sents.append(len(word_idxs))
#                    word_idxs.extend(find_spans(text[offset:], 
#                                                [node["word"] 
#                                                 for node in sent.nodes.values()
#                                                 if node["word"] is not None], 
#                                                offset))
#                    deps.extend([(node.head(), node.rel()) 
#                                 for node in sent.nodes])
#                    offset += len(detokenizer.detokenize(sent))
#                    offset += 1
#                offset += 2
#            doc.words = word_idxs
#            doc.sentences = sents
#            doc.paragraphs = paras
#            doc.dep = deps
#        elif sentences:
#            text = " ".join(detokenizer.detokenize(sent)
#                            for sent in nltk_corpus.sents(fileid))
#            doc = corpus.add_doc(text)
#            offset = 0
#            sents = []
#            word_idxs = []
#            deps = []
#            for sent in nltk_corpus.parsed_sents(fileid):
#                sents.append(len(word_idxs))
#                word_idxs.extend(find_spans(text[offset:], 
#                                            [node["word"] 
#                                             for node in sent.nodes.values()
#                                             if node["word"] is not None],
#                                            offset))
#                deps.extend([(node.head(), node.rel()) 
#                             for node in sent.nodes])
#                offset += len(detokenizer.detokenize(sent))
#                offset += 1
#            doc.words = word_idxs
#            doc.sentences = sents
#            doc.dep = deps
#        else:
#            raise ValueError("Cannot convert corpus without sentences")
#
#        doc.tags = [tag for (word, tag) in nltk_corpus.tagged_words(fileid)]
#    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
#        corpus.to_yaml(f)
#

def convert_one_tree(tree, parent=0, n=0):
    if all(isinstance(subtree, tuple) for subtree in tree):
        return ([subtree[0] for subtree in tree], 
                [(n + i, n + i + 1, subtree[1]) for i, subtree in enumerate(tree)],
                [(parent, n + i) for i, subtree in enumerate(tree)])
    if len(tree) == 1 and isinstance(tree[0], str):
        return [tree[0]], [(n, n+1, tree.label())], [(parent, n)]
    words = []
    nodes = []
    constituents = []
    orig_n = n
    constituents.append((parent, n))
    parent = n
    n += 1
    for subtree in tree:
        w, ns, i = convert_one_tree(subtree, parent, n)
        words.extend(w)
        nodes.extend(ns)
        constituents.extend(i)
        n += len(ns)
    nodes.append((orig_n, n, tree.label()))
        
    return words, nodes, constituents

def convert_tree_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("nodes", layer_type="span", base="words",
                          data="string")
    corpus.add_layer_meta("constituents", layer_type="element", base="nodes",
                          target="nodes", data="link")
    sents = False
    try:
        nltk_corpus.sents()
        corpus.add_layer_meta("sentences", layer_type="div", base="words")
        sentences = True
    except Exception:
        pass

    paragraphs = False
    try:
        nltk_corpus.paras()
        corpus.add_layer_meta("paragraphs", layer_type="div", base="sentences")
        paragraphs = True
    except Exception:
        pass
    for fileid in nltk_corpus.fileids():
        if paragraphs:
            text = "\n\n".join(" ".join(
                detokenizer.detokenize(sent) 
                for sent in para) 
                for para in nltk_corpus.paras(fileid))
            doc = corpus.add_doc(text)
            offset = 0
            sents = []
            paras = []
            word_idxs = []
            nodes = []
            constituents = []
            for para in nltk_corpus.parsed_paras(fileid):
                paras.append(len(sents))
                for sent in para:
                    sents.append(len(word_idxs))
                    w, n, c = convert_one_tree(sent, len(nodes), len(nodes))
                    word_idxs.extend(find_spans(text[offset:], w, offset))
                    nodes.extend(n)
                    constituents.extend(c)
                    offset += len(detokenizer.detokenize(w))
                    offset += 1
                offset += 2
            doc.words = word_idxs
            doc.sentences = sents
            doc.paragraphs = paras
            doc.nodes = nodes
            doc.constituents = constituents
        elif sentences:
            text = " ".join(detokenizer.detokenize(sent)
                            for sent in nltk_corpus.sents(fileid))
            doc = corpus.add_doc(text)
            offset = 0
            sents = []
            word_idxs = []
            nodes = []
            constituents = []
            print(fileid)
            for sent in nltk_corpus.parsed_sents(fileid):
                sents.append(len(word_idxs))
                w, n, c = convert_one_tree(sent, len(nodes), len(nodes))
                word_idxs.extend(find_spans(text[offset:], w, offset))
                nodes.extend(n)
                constituents.extend(c)
                offset += len(detokenizer.detokenize(w))
                offset += 1
            doc.words = word_idxs
            doc.sentences = sents
            doc.nodes = nodes
            doc.constituents = constituents
        else:
            raise "Unreachable"

    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

 

#convert_plain_text("abc")
#
#convert_tagged_corpus("alpino")
#
#convert_tagged_corpus("brown")

# TODO: cess_cat, cess_esp 
convert_tree_corpus("cess_cat")

# TODO: chat80

# TODO: comparative_sentences

# TODO: comtrans

#convert_tagged_corpus("conll2000")
#
#convert_tagged_corpus("conll2002")

#convert_dep_corpus("conll2007")
# TODO: dependency_treebank

# TODO: 
