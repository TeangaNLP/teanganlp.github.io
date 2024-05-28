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
    last_start = 0
    for token in tokens:
        if token == '``':
            token = "\""
        elif token == "''":
            token = "\""
        elif token == ". ...":
            token = "...."
        elif token.startswith("( ") and token.endswith(" )"):
            token = "(" + token[2:-2] + ")"
        else:
            if "''" in token:
                token = token.replace("''", "\"")
            if "``" in token:
                token = token.replace("``", "\"")
        last_start = start
        start = text.find(token, start)
        if start == -1:
            raise ValueError("Token not found in text: " + token + 
                             " in " + text[last_start:last_start+50])
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
    sentences = False
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
            doc.fileid = [fileid]
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
                offset += 1
            doc.words = word_idxs
            doc.sentences = sents
            doc.paragraphs = paras
        elif sentences:
            text = " ".join(detokenizer.detokenize(sent)
                            for sent in nltk_corpus.sents(fileid))
            doc = corpus.add_doc(text)
            doc.fileid = [fileid]
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

def convert_one_tree(tree, parent=0, n=0, nw=0):
    if all(isinstance(subtree, tuple) for subtree in tree):
        return ([subtree[0] for subtree in tree], 
                [(n + i, n + i + 1, subtree[1]) for i, subtree in enumerate(tree)],
                [(parent, n + i) for i, subtree in enumerate(tree)])
    if len(tree) == 1 and isinstance(tree[0], str):
        return [tree[0]], [(n, n+1, tree.label())], [(parent, n)]
    words = []
    nodes = []
    constituents = []
    constituents.append((parent, n))
    parent = n
    n += 1
    for subtree in tree:
        w, ns, i = convert_one_tree(subtree, parent, n, len(words) + nw)
        words.extend(w)
        nodes.extend(ns)
        constituents.extend(i)
        n += len(ns)
    nodes.insert(0,(nw, nw + len(words), tree.label()))
        
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
            doc.fileid = [fileid]
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
                    w, n, c = convert_one_tree(sent, len(nodes), len(nodes), 
                                               len(word_idxs))
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
            doc.fileid = [fileid]
            offset = 0
            sents = []
            word_idxs = []
            nodes = []
            constituents = []
            for sent in nltk_corpus.parsed_sents(fileid):
                sents.append(len(word_idxs))
                w, n, c = convert_one_tree(sent, len(nodes), len(nodes), len(word_idxs))
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

#convert_tree_corpus("cess_cat")
#convert_tree_corpus("cess_esp")

def convert_comparative_sentences(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("comp_type", layer_type="seq", base="document", 
                          data="string")
    corpus.add_layer_meta("entity_1", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("entity_2", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("feature", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("keyword", layer_type="seq", base="document",
                          data="string")
    for comparison in nltk_corpus.comparisons():
        text = detokenizer.detokenize(comparison.text)
        doc = corpus.add_doc(text)
        doc.words = find_spans(text, comparison.text)
        if comparison.comp_type:
            doc.comp_type = [comparison.comp_type]
        if comparison.entity_1:
            doc.entity_1 = [comparison.entity_1]
        if comparison.entity_2:
            doc.entity_2 = [comparison.entity_2]
        if comparison.feature:
            doc.feature = [comparison.feature]
        if comparison.keyword:
            doc.keyword = [comparison.keyword]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)
 
#convert_comparative_sentences("comparative_sentences")    

# TODO: comtrans
def convert_aligned_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text1")
    corpus.add_layer_meta("text2")
    corpus.add_layer_meta("document", layer_type="div", base="text1", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("words1", layer_type="span", base="text1")
    corpus.add_layer_meta("words2", layer_type="span", base="text2")
    corpus.add_layer_meta("alignment", layer_type="element", base="words1",
                          target="words2", data="link")
    for fileid in nltk_corpus.fileids():
        for sentence in nltk_corpus.aligned_sents(fileid):
            text1 = detokenizer.detokenize(sentence.words)
            text2 = detokenizer.detokenize(sentence.mots)
            doc = corpus.add_doc(text1=text1, text2=text2)
            doc.words1 = find_spans(text1, sentence.words)
            doc.words2 = find_spans(text2, sentence.mots)
            doc.alignment = [(i, j) for i, j in sentence.alignment]
            doc.fileid = [fileid]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)
 
#convert_aligned_corpus("comtrans")

#convert_tagged_corpus("conll2000")

#convert_tagged_corpus("conll2002")

#convert_dep_corpus("conll2007")

#convert_tagged_corpus("floresta")

#convert_plain_text("genesis")

#convert_plain_text("gutenberg")

def convert_ieer_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("entity", layer_type="span", base="words",
                          data=list(set(
                              w.label()
                              for doc in nltk_corpus.parsed_docs()
                              for w in doc.text
                              if isinstance(w, nltk.tree.Tree))))
    for nltk_doc in nltk_corpus.parsed_docs():
        words = []
        entities = []
        for word in nltk_doc.text:
            if isinstance(word, nltk.tree.Tree):
                entities.append((len(words), 
                                 len(words) + len(word.leaves()),
                                 word.label()))
                words.extend(word.leaves())
            else:
                words.append(word)
        text = detokenizer.detokenize(words)
        doc = corpus.add_doc(text)
        doc.words = find_spans(text, words)
        doc.entity = entities
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

#convert_ieer_corpus("ieer")

#convert_plain_text("inaugural")

#convert_tagged_corpus("indian")

#convert_plain_text("machado")

#convert_tagged_corpus("mac_morpho")

#convert_tagged_corpus("masc_tagged")

#convert_plain_text("movie_reviews")

#convert_tagged_corpus("nps_chat")

def convert_review_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("title", layer_type="seq", base="document")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("sentences", layer_type="div", base="words")
    corpus.add_layer_meta("features", layer_type="element", base="sentences",
                          data="string")
    corpus.add_layer_meta("notes", layer_type="element", base="sentences",
                          data="string")
    for review in nltk_corpus.reviews():
        words = []
        features = []
        sentences = []
        notes = []
        text = ""
        sent_no = 0
        for review_line in review.review_lines:
            sentences.append(len(words))
            n = len(text)
            text = text + detokenizer.detokenize(review_line.sent)
            words.extend(find_spans(text, review_line.sent, n))
            features.extend([(sent_no, f"{f[0]}={f[1]}") for f in review_line.features])
            notes.extend([(sent_no, note) for note in review_line.notes])
            sent_no += 1
        doc = corpus.add_doc(text)
        doc.title = [review.title]
        doc.words = words
        doc.sentences = sentences
        doc.features = features
        doc.notes = notes
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

#convert_review_corpus("product_reviews_1")
#convert_review_corpus("product_reviews_2")

#convert_plain_text("pros_cons")

def convert_string_category_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("category", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    for fileid in nltk_corpus.fileids():
        for label, text in nltk_corpus.tuples(fileid):
            doc = corpus.add_doc(text)
            doc.category = [label]
            doc.fileid = [fileid]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

#convert_string_category_corpus("qc")

def convert_rte_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("hyp")
    corpus.add_layer_meta("document", layer_type="div", base="text", default=[0])
    corpus.add_layer_meta("fileid", layer_type="seq", base="document")
    corpus.add_layer_meta("challenge", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("id", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("value", layer_type="seq", base="document",
                          data="string")
    corpus.add_layer_meta("task", layer_type="seq", base="document",
                          data="string")
    for fileid in nltk_corpus.fileids():
        for pair in nltk_corpus.pairs(fileid):
            doc = corpus.add_doc(text=pair.text, hyp=pair.hyp)
            doc.fileid = [fileid]
            doc.challenge = [pair.challenge]
            doc.id = [pair.id]
            doc.value = [str(pair.value)]
            doc.task = [pair.task]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

convert_rte_corpus("rte")
