"""q2 (T4b): the A-lens question set. Wording 0 of q3..q7 = scripts/q1/q1_lib2.NEW_Q verbatim (the D1 prompts);
wordings 1, 2 written for q2 in the same style (chat few-shot, bare label prefilled, 4 examples in mixed languages,
each with a tempting answer at the wrong level; golds checked against runs/sub1/exclude_concepts.txt, the probe set
and the passage text at run time). Format per wording: (header, query prefix, label, [(text, gold)]).
Existing questions for reference: topic = scripts/ask3/ask3_collect9.W, spec (unstated word) = scripts/ask5/ask5_collect.W.
"""
import sys

sys.path.insert(0, "/workspace/o-lens-improvement/scripts/q1")

NEW = {
    "q3": [
        ("", "Text: ", "The person or thing referred to:", [
            ("The Dutch master of light and shadow painted The Night Watch in Amsterdam.", "Rembrandt"),  # painting
            ("La reina que gobernó Egipto y se enamoró de Marco Antonio murió por la mordedura de un áspid.",
             "Cleopatra"),                                                                             # Egipto
            ("It has the longest neck of any animal and eats from the tops of acacia trees.", "giraffe"),  # neck
            ("The first human to orbit the Earth, in 1961, came from a small village near Smolensk.", "Gagarin")]),
        ("", "Passage: ", "Name the one being described:", [
            ("The Austrian monk who bred pea plants in his monastery garden founded the study of heredity.", "Mendel"),
            ("Il dort le jour, chasse la nuit et tourne la tête presque complètement.", "hibou"),         # nuit
            ("The giant German airship burst into flames while landing in New Jersey in 1937.", "Hindenburg"),
            ("It whistles on the stove when the water is ready.", "kettle")]),                          # tea
    ],
    "q4": [
        ("", "Text: ", "The hidden step to figure out first (one word):", [
            ("Die Farbe des Blattes auf der Flagge des Landes nördlich der USA ist", "Kanada"),     # tempting: rot
            ("The instrument played by the composer of The Four Seasons is the", "Vivaldi"),       # violin
            ("The currency of the country whose capital is Tbilisi is the", "Georgia"),           # lari
            ("Ein Dutzend Eier minus zwei sind", "zwölf")]),                                       # zehn
        ("", "Passage: ", "First you need to know (one word):", [
            ("The river flowing through the capital of the Czech Republic is the", "Prague"),     # Vltava
            ("La capitale du pays où est né Chopin s'appelle", "Pologne"),                         # Varsovie
            ("The language of the playwright who wrote Faust is", "Goethe"),                       # German
            ("The author of Don Quixote wrote in the language of", "Cervantes")]),                 # Spanish
    ],
    "q5": [
        ("", "Text: ", "Correctly spelled, the final word is:", [
            ("pls dont forget the documnts", "documents"),
            ("The recipe calls for two cups of buttermilck", "buttermilk"),
            ("Wir trafen uns am Banhof", "Bahnhof"),
            ("lol that was so funy", "funny")]),
        ("", "Passage: ", "Fix the spelling of the last word:", [
            ("He finally got his licence to drive a tracktor", "tractor"),
            ("brb gotta grab my jackit", "jacket"),
            ("Ella vive en la ciudad de Barselona", "Barcelona"),
            ("The museum had a huge skeletn", "skeleton")]),
    ],
    "q6": [
        ("", "Text: ", "A word that is coming up soon:", [
            ("The barista ground the beans, steamed the milk and drew a heart in the", "foam"),        # coffee
            ("She tore the wrapping paper, read the card and blew out the", "candles"),               # birthday
            ("Der Schiedsrichter pfiff, der Stürmer lief an und schoss den", "Ball"),                  # Fußball
            ("To plant the seeds, first dig a small hole, then cover them with", "soil")]),            # garden
        ("", "Passage: ", "Later the text will mention:", [
            ("The girl kissed the slimy green creature by the pond, hoping it would turn into a", "prince"),  # frog
            ("The pilot announced the descent, the seatbelt sign came on and the wheels touched the", "runway"),
            ("Ella mezcló la harina con agua y levadura y dejó reposar la masa para hacer", "pan"),
            ("The keeper dove to his left, the striker smiled, and the ball hit the back of the net for a",
             "goal")]),
    ],
    "q7": [
        ("", "Text: ", "It makes you think of:", [
            ("A red nose, oversized shoes and a tiny car that holds twenty people.", "circus"),        # clown
            ("Snow White, a poisoned fruit and a mirror on the wall.", "Grimm"),                         # apple
            ("Fjords, Vikings and long winter nights.", "Scandinavia"),                                  # cold
            ("Un sombrero de copa, una varita y un conejo blanco.", "mago")]),                          # conejo
        ("", "Passage: ", "What it calls to mind:", [
            ("A lamp rubbed three times and a blue giant offering three wishes.", "genie"),             # lamp
            ("A black cat, a broken mirror and a ladder you walk under.", "superstition"),              # cat
            ("Il pleut, les feuilles tombent, rouges et jaunes, et les jours raccourcissent.", "octobre"),
            ("Spinach, a pipe and huge forearms.", "Popeye")]),                                          # sailor
    ],
}
NAME = {"q3": "q3_referent", "q4": "q4_intermediate", "q5": "q5_canonical", "q6": "q6_plan", "q7": "q7_association"}


def specs():
    """{q: [w0, w1, w2]} with w0 = q1_lib2.NEW_Q (imported lazily: q1_lib2 imports torch)."""
    import q1_lib2 as Q
    return {q: [Q.NEW_Q[NAME[q]]] + NEW[q] for q in NEW}


def chat(tok, user, prefill):  # = q1_lib2.chat = ask5_collect.build(FAM="chat")
    msgs = [{"role": "user", "content": user}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    if "</think>" not in s.split("<|im_start|>assistant")[-1]:
        s += "<think>\n\n</think>\n\n"
    return s + prefill


def prompt(tok, spec, text):
    head, q, lab, ex = spec
    shots = head + "".join(f"{q}{t}\n{lab} {w}\n\n" for t, w in ex)
    return chat(tok, shots + f"{q}{text}\n{lab}", lab)


def hygiene(R="/workspace/o-lens-improvement"):
    import json
    E = {l.strip().lower() for l in open(f"{R}/runs/sub1/exclude_concepts.txt")}
    P = json.load(open(f"{R}/probes/probes_v1.json"))["probes"]
    pc = {x.lower() for p in P for x in p["accepted"] + p.get("near", []) if isinstance(x, str)}
    for q, ws in NEW.items():
        for k, (_, _, lab, ex) in enumerate(ws, 1):
            for t, w in ex:
                assert w.lower() not in t.lower(), (q, k, w)
                assert w.lower() not in E and w.lower() not in pc, (q, k, w, "listed concept")
    print("[q2 qs] hygiene ok: new wordings' golds not in passages, not listed bench concepts, not probe words")
