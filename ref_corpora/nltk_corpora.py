import nltk
import teanga
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import json

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
    corpus.add_layer_meta("fileid")
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    for fileid in nltk_corpus.fileids():
        text = nltk_corpus.raw(fileid)
        doc = corpus.add_doc(text=text)
        doc.fileid = fileid
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

def convert_tagged_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("fileid")
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
            doc = corpus.add_doc(text=text)
            doc.fileid = fileid
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
            doc = corpus.add_doc(text=text)
            doc.fileid = fileid
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
            doc = corpus.add_doc(text=text)
            doc.fileid = [fileid]
            doc.words = find_spans(text, words)

        doc.tags = [tag for (word, tag) in nltk_corpus.tagged_words(fileid)]
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

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

def convert_tree_corpus(corpus_name, simple_detokenize=False):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("fileid")
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
            if simple_detokenize:
                text = "\n\n".join(" ".join(
                    "".join(sent)
                    for sent in para)
                    for para in nltk_corpus.paras(fileid))
            else:
                text = "\n\n".join(" ".join(
                    detokenizer.detokenize(sent) 
                    for sent in para) 
                    for para in nltk_corpus.paras(fileid))
            doc = corpus.add_doc(text=text)
            doc.fileid = fileid
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
                    if simple_detokenize:
                        offset += len("".join(w))
                    else:
                        offset += len(detokenizer.detokenize(w))
                    offset += 1
                offset += 2
            doc.words = word_idxs
            doc.sentences = sents
            doc.paragraphs = paras
            doc.nodes = nodes
            doc.constituents = constituents
        elif sentences:
            if simple_detokenize:
                text = " ".join("".join(sent) for sent in nltk_corpus.sents(fileid))
            else:
                text = " ".join(detokenizer.detokenize(sent)
                                for sent in nltk_corpus.sents(fileid))
            doc = corpus.add_doc(text=text)
            doc.fileid = fileid
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
                if simple_detokenize:
                    offset += len("".join(w))
                else:
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

convert_plain_text("abc")

convert_tagged_corpus("alpino")

convert_tagged_corpus("brown")

convert_tree_corpus("cess_cat")
convert_tree_corpus("cess_esp")

def convert_comparative_sentences(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("comp_type")
    corpus.add_layer_meta("entity_1")
    corpus.add_layer_meta("entity_2")
    corpus.add_layer_meta("feature")
    corpus.add_layer_meta("keyword")
    for comparison in nltk_corpus.comparisons():
        text = detokenizer.detokenize(comparison.text)
        doc = corpus.add_doc(text=text)
        doc.words = find_spans(text, comparison.text)
        if comparison.comp_type:
            doc.comp_type = comparison.comp_type
        if comparison.entity_1:
            doc.entity_1 = comparison.entity_1
        if comparison.entity_2:
            doc.entity_2 = comparison.entity_2
        if comparison.feature:
            doc.feature = comparison.feature
        if comparison.keyword:
            doc.keyword = comparison.keyword
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)
 
convert_comparative_sentences("comparative_sentences")    

def convert_aligned_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text1")
    corpus.add_layer_meta("text2")
    corpus.add_layer_meta("fileid")
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
            doc.fileid = fileid
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)
 
convert_aligned_corpus("comtrans")

convert_tagged_corpus("conll2000")

convert_tagged_corpus("conll2002")

convert_tagged_corpus("floresta")

convert_plain_text("genesis")

convert_plain_text("gutenberg")

def convert_ieer_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
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
        doc = corpus.add_doc(text=text)
        doc.words = find_spans(text, words)
        doc.entity = entities
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

convert_ieer_corpus("ieer")

convert_plain_text("inaugural")

convert_tagged_corpus("indian")

convert_plain_text("machado")

convert_tagged_corpus("mac_morpho")

convert_tagged_corpus("masc_tagged")

convert_plain_text("movie_reviews")

convert_tagged_corpus("nps_chat")

def convert_review_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("title")
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
        doc = corpus.add_doc(text=text)
        doc.title = review.title
        doc.words = words
        doc.sentences = sentences
        doc.features = features
        doc.notes = notes
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

convert_review_corpus("product_reviews_1")
convert_review_corpus("product_reviews_2")

convert_plain_text("pros_cons")

def convert_string_category_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("category")
    corpus.add_layer_meta("fileid")
    for fileid in nltk_corpus.fileids():
        for label, text in nltk_corpus.tuples(fileid):
            doc = corpus.add_doc(text=text)
            doc.category = label
            doc.fileid = fileid
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

convert_string_category_corpus("qc")

def convert_rte_corpus(corpus_name):
    corpus = teanga.Corpus()
    nltk_corpus = eval("nltk.corpus." + corpus_name)
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("hyp")
    corpus.add_layer_meta("fileid")
    corpus.add_layer_meta("challenge")
    corpus.add_layer_meta("id")
    corpus.add_layer_meta("value")
    corpus.add_layer_meta("task")
    for fileid in nltk_corpus.fileids():
        for pair in nltk_corpus.pairs(fileid):
            doc = corpus.add_doc(text=pair.text, hyp=pair.hyp)
            doc.fileid = fileid
            doc.challenge = pair.challenge
            doc.id = pair.id
            doc.value = str(pair.value)
            doc.task = pair.task
    with open("../corpora/" + corpus_name + ".yaml", "w") as f:
        corpus.to_yaml(f)

convert_rte_corpus("rte")

def convert_senseval():
    corpus = teanga.Corpus()
    nltk_corpus = nltk.corpus.senseval
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("tags", layer_type="seq", base="words",
                          data=['', '"', '#', '$', "''", '(', ')', ',', '.', ':', 
                                'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 
                                'JJR', 'JJS', 'JJ|JJR', 'MD', 'NN', 'NNP', 
                                'NNPS', 'NNP|JJ', 'NNS', 'NN|DT', 'NN|JJ', 
                                'PDT', 'POS', 'PRP', 'PRP$', 'R', 'RB', 'RBR', 
                                'RBS', 'RP', 'S', 'SYM', 'TO', 'UH', 'VB', 
                                'VBD', 'VBG', 'VBG|NN', 'VBN', 'VBP', 'VBZ', 
                                'WDT', 'WP', 'WP$', 'WRB'])
    corpus.add_layer_meta("word")
    corpus.add_layer_meta("senseid")
    corpus.add_layer_meta("head", layer_type="element", base="words")
    for instance in nltk_corpus.instances():
        words = []
        tags = []
        for w in instance.context:
            if isinstance(w, tuple):
                words.append(w[0])
                tags.append(w[1])
            else:
                words.append(w)
                tags.append("")
        text = detokenizer.detokenize(words)
        doc = corpus.add_doc(text=text)
        doc.words = find_spans(text, words)
        doc.tags = tags
        doc.word = instance.word
        doc.senseid = [instance.senses[0]]
        doc.head = [(instance.position,)]
    with open("../corpora/senseval.yaml", "w") as f:
        corpus.to_yaml(f)

convert_senseval()

convert_plain_text("sentence_polarity")
convert_plain_text("shakespeare")

# NLTK bugs?
#convert_tree_corpus("sinica_treebank", simple_detokenize=True)

convert_plain_text("state_union")

convert_plain_text("subjectivity")

def convert_switchboard():
    corpus = teanga.Corpus()
    nltk_corpus = nltk.corpus.switchboard
    corpus.add_layer_meta("text")
    corpus.add_layer_meta("words", layer_type="span", base="text")
    corpus.add_layer_meta("tags", layer_type="seq", base="words",
                          data=[',', '.', ':', 'BES', 'CC', 'CD', 'DT', 'EX', 
                                'FW', 'GW', 'HVS', 'IN', 'JJ', 'JJR', 'JJS', 
                                'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS',
                                'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
                                'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 
                                'VBZ', 'WDT', 'WP', 'WP$', 'WRB', 'XX', '^CC',
                                '^DT', '^IN', '^JJ', '^NN', '^NNS', '^NNS^POS',
                                '^NN^BES', '^NN^POS', '^PRP$', '^PRP^BES', 
                                '^PRP^VBP', '^RB', '^VB', '^VBD', '^VBG', 
                                '^VBN', '^VBP', '^VB^RP', '^WP^BES'])
    corpus.add_layer_meta("speaker")
    corpus.add_layer_meta("id")
    for turn in nltk_corpus.tagged_turns():
        words = [t[0] for t in turn]
        tags = [t[1] for t in turn]
        text = detokenizer.detokenize(words)
        doc = corpus.add_doc(text=text)
        doc.words = find_spans(text, words)
        doc.tags = tags
        doc.speaker = turn.speaker
        doc["id"] = str(turn.id)
    with open("../corpora/switchboard.yaml", "w") as f:
        corpus.to_yaml(f)

convert_switchboard()

convert_tree_corpus("treebank")

def convert_twitter_corpus():
    corpus = teanga.Corpus()
    twitter_corpus = nltk.corpus.twitter_samples
    for key in ['contributors', 'coordinates', 'created_at', 'entities', 
                'extended_entities', 'favorite_count', 'favorited',
                'filter_level', 'geo', 'id', 'id_str',
                'in_reply_to_screen_name', 'in_reply_to_status_id',
                'in_reply_to_status_id_str', 'in_reply_to_user_id',
                'in_reply_to_user_id_str', 'is_quote_status', 'lang', 
                'metadata', 'place', 'possibly_sensitive', 'quoted_status',
                'quoted_status_id', 'quoted_status_id_str', 'retweet_count',
                'retweeted', 'retweeted_status', 'source', 'text', 
                'timestamp_ms', 'truncated', 'user']:
        corpus.add_layer_meta(key)
    corpus.add_layer_meta("fileid")
    for fileid in twitter_corpus.fileids():
        for tweet in twitter_corpus.docs(fileid):
            doc = corpus.add_doc(text=tweet["text"])
            for key in tweet:
                if tweet[key] is None:
                    pass
                elif isinstance(tweet[key], str):
                    doc[key] = [tweet[key]]
                else:
                    doc[key] = json.dumps(tweet[key])
            doc.fileid = fileid
    with open("../corpora/twitter_samples.yaml", "w") as f:
        corpus.to_yaml(f)

convert_twitter_corpus()

convert_plain_text("udhr")
convert_plain_text("udhr2")

convert_plain_text("webtext")
