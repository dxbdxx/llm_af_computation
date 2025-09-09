import os
import random
import copy
import json
import pickle
from multiprocessing import Pool, cpu_count
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm

from src import config


class JsonEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super(JsonEncoder, self).__init__(*args, **kwargs)
        self.indent = kwargs.get('indent', None)

    def encode(self, o):
        if isinstance(o, dict):
            items = [f'\n{" " * self.indent}"{k}": {self.encode(v)}' for k, v in o.items()]
            return '{' + ','.join(items) + '\n}'
        elif isinstance(o, list):
            return '[' + ', '.join(self.encode(i) for i in o) + ']'
        else:
            return super(JsonEncoder, self).encode(o)


def draw_graph(graph, save=False):
    pos = nx.circular_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=10, font_color="black",
            font_weight="bold", arrowsize=20)
    if save:
        plt.savefig('temp1.png')
    else:
        plt.show()


def get_attacker(G, arg):
    return set([a for a, b in G.edges if b == arg])


def get_attacked(G, arg):
    return set([b for a, b in G.edges if a == arg])


def get_instruction_string_grd(G):
    s1 = ""
    s1 += "We are solving the grounded extension of an abstract argumentation framework AF which is described by a directed graph. "

    if random.random() > 0.5:
        s1 += "In the graph, the numbers are arguments and the edge means attack relation between arguments. "
    if random.random() > 0.5:
        s1 += "The grounded extension is a set of arguments, which is the minimum set of complete extensions. "
    if random.random() > 0.5:
        s1 += "A set is a complete extension if and only if it is conflict-free, it defends all its arguments and it contains all the arguments it defends. "

    s1 += "Let's solve this computation problem step by step and answer the problem with a json object."

    s2 = ""
    c = random.choice([1, 2, 3])
    if c == 1:
        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
        dot_string = pydot_graph.to_string()
        s2 += f"The AF is described by following graph in dot format:\n{dot_string}\n"
    elif c == 2:
        graphml_str = '\n'.join(nx.generate_graphml(G))
        s2 += f"The AF is described by following graph in graphml format:\n{graphml_str}\n"
    elif c == 3:
        data = nx.node_link_data(G)
        json_str = json.dumps(data, indent=4)
        s2 += f"The AF is described by following graph in json format:\n{json_str}\n"

    return s1, s2


def grd_section(G, exts, json_dir, f):
    args = set(G.nodes)
    ext_grd = set(list(exts['grd'])[0])
    ext_grd_out = set()
    for arg in ext_grd:
        attackeds = get_attacked(G, arg)
        ext_grd_out.update(attackeds)
    ext_grd_und = args - ext_grd - ext_grd_out

    prompt_dir = json_dir / "prompt__grd"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    file_name = f.stem
    with open(prompt_dir / (file_name + "_grd" + ".txt"), 'w') as f_write:
        # ---------- question
        inst_string, input_string = get_instruction_string_grd(G)
        print("instruction:" + inst_string, file=f_write)
        print("input:" + input_string, file=f_write)

        # ---------- explanation
        set_in = set()
        set_out = set()
        set_rest = copy.deepcopy(args)

        # ---------- init
        print("output:Unattacked arguments are legally IN.", file=f_write)
        print("First find unattacked arguments.", file=f_write)
        grd_empty = not any(len(get_attacker(G, arg)) == 0 for arg in copy.deepcopy(set_rest))
        if grd_empty:
            print("All arguments are attacked, so the grounded extension is the empty set.", file=f_write)
            answer_section_grd(ext_grd, ext_grd_out, ext_grd_und, f_write)
            return

        print("======== STEP IN =========", file=f_write)
        for arg in copy.deepcopy(set_rest):
            attackers = get_attacker(G, arg)
            if len(attackers) == 0:
                print(f"{arg} is not attacked, label it IN.", file=f_write)
                set_in.add(arg)
                set_rest.remove(arg)

        if len(set_rest) == len(ext_grd_und):
            answer_section_grd(ext_grd, ext_grd_out, ext_grd_und, f_write)
            return
        print("======== STEP END =========", file=f_write)

        print("\nThen we recursively find the grounded labelling.", file=f_write)
        IN_OUT_recursion(set_in, set_out, set_rest, ext_grd, ext_grd_out, ext_grd_und, G, f_write)


def IN_OUT_recursion(set_in: set, set_out: set, set_rest: set, ext_grd, ext_grd_out, ext_grd_und, G, f_write):
    print("======== STEP OUT =========", file=f_write)
    for arg in set_in:
        attackeds = get_attacked(G, arg)
        for arg1 in attackeds:
            if arg1 not in set_out:
                print(f"{arg1} is attacked by {arg} which is labelled IN, N, so {arg1} can be legally labelled OUT;",
                      file=f_write)
                set_out.add(arg1)
                set_rest.remove(arg1)
    print("======== STEP END =========", file=f_write)

    if len(set_rest) == len(ext_grd_und):
        answer_section_grd(ext_grd, ext_grd_out, ext_grd_und, f_write)
        return

    print("======== STEP IN =========", file=f_write)
    for arg in copy.deepcopy(set_rest):
        attackers = get_attacker(G, arg)
        diff = attackers - set_out
        if len(diff) == 0:
            if len(attackers) == 1:
                arg1 = list(attackers)[0]
                print(f"{arg} is attacked by {arg1} which is labelled OUT, because ", end="", file=f_write)
                attackers1 = get_attacker(G, arg1)
                att_options = []
                for att in attackers1:
                    if att in set_in:
                        att_options.append(att)
                att = random.choice(att_options)
                print(f"{arg1} is attacked by {att} which is labelled IN.", file=f_write)
            else:
                print(f"{arg} is attacked by {attackers} while they are labelled OUT, because ", end="", file=f_write)
                str_attack = ""
                for arg1 in attackers:
                    attackers1 = get_attacker(G, arg1)
                    att_options = []
                    for att in attackers1:
                        if att in set_in:
                            att_options.append(att)

                    att = random.choice(att_options)
                    str_attack += f"{arg1} is attacked by {att} which is labelled IN; "

                print(str_attack[:-2] + '.', file=f_write)

            set_in.add(arg)
            set_rest.remove(arg)

    print("======== STEP END =========", file=f_write)

    if len(set_rest) == len(ext_grd_und):
        answer_section_grd(ext_grd, ext_grd_out, ext_grd_und, f_write)
        return

    IN_OUT_recursion(set_in, set_out, set_rest, ext_grd, ext_grd_out, ext_grd_und, G, f_write)


def answer_section_grd(set_in, set_out, set_und, f_write):
    # ---------- answer
    ans_dict = {"IN": sorted(list(set_in)), "OUT": sorted(list(set_out)), "UNDEC": sorted(list(set_und))}
    json_string = f"\nFinally the answer in json format is {json.dumps(ans_dict, cls=JsonEncoder, indent=4)}"
    print(json_string, file=f_write)


def get_instruction_string_com(G, set_in, set_out, set_und):
    s1 = ""
    s1 += "We are solving complete extensions of an abstract argumentation framework AF which is described by a directed graph. "

    if random.random() > 0.5:
        s1 += "In the graph, the numbers are arguments and the edge means attack relation between arguments. "
    if random.random() > 0.5:
        s1 += "A set of arguments is a complete extension if and only if it is conflict-free, it defends all its arguments and it contains all the arguments it defends. "

    # s1 += "Let's solve this computation problem step by step and answer the problem with a json object."
    s1 += f"The grounded labelling is IN: {set_in}, OUT: {set_out}, UNDEC: {set_und}. "
    s1 += "In complete labellings, IN set is conflict-free, it defends all its arguments and it contains all the arguments it defends."
    s2 = ""

    c = random.choice([1, 2, 3])
    if c == 1:
        pydot_graph = nx.drawing.nx_pydot.to_pydot(G)
        dot_string = pydot_graph.to_string()
        s2 += f"The AF is described by following graph in dot format:\n{dot_string}\n"
    elif c == 2:
        graphml_str = '\n'.join(nx.generate_graphml(G))
        s2 += f"The AF is described by following graph in graphml format:\n{graphml_str}\n"
    elif c == 3:
        data = nx.node_link_data(G)
        json_str = json.dumps(data, indent=4)
        s2 += f"The AF is described by following graph in json format:\n{json_str}\n"

    if random.random() > 0.5:
        s2 += "The grounded extension is the minimum complete extension."

    return s1, s2


def com_section(G, exts, json_dir, f):
    args = set(G.nodes)
    ext_grd = set(list(exts['grd'])[0])
    ext_com = [set(e) for e in exts['com']]
    ext_grd_out = set()
    for arg in ext_grd:
        attackeds = get_attacked(G, arg)
        ext_grd_out.update(attackeds)
    ext_grd_und = args - ext_grd - ext_grd_out

    prompt_dir = json_dir / "prompt__com"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    file_name = f.stem
    with open(prompt_dir / (file_name + "_com" + ".txt"), 'w') as f_write:
        # ---------- question
        set_in = copy.deepcopy(ext_grd)
        set_out = copy.deepcopy(ext_grd_out)
        set_und = copy.deepcopy(ext_grd_und)

        inst_string, input_string = get_instruction_string_com(G, set_in, set_out, set_und)
        print("instruction:" + inst_string, file=f_write)
        print("input:" + input_string, file=f_write)
        print("output:", file=f_write, end="")
        # ---------- init
        set_ins = []
        set_outs = []
        set_unds = []

        set_ins.append(set_in)
        set_outs.append(set_out)
        set_unds.append(set_und)

        # ---------- explanation
        print(f"We already know that the grounded labelling is IN: {set_in}, OUT: {set_out}, UNDEC: {set_und}.",
              file=f_write)
        print(f"Let’s try to find all other complete labellings.", file=f_write)

        if len(set_und) == 0:
            print("UNDEC set is empty, so the sole complete extension is the grounded extenstion.", file=f_write)
            answer_section_com(set_ins, set_outs, set_unds, f_write)
            return

        # print(f"So let's add IN set some elements of UNDEC: {set_und} to find all other complete extensions.",
        #       file=f_write)
        if len(ext_com) == 1:
            print(
                "We cannot find other complete extensions, so the sole complete extension is the grounded extenstion.",
                file=f_write)
            answer_section_com(set_ins, set_outs, set_unds, f_write)
            return

        for e_com in ext_com:
            if e_com != set_in:
                diff1 = e_com - set_in
                set_in1 = e_com
                set_out1 = set()
                for arg in set_in1:
                    attackeds = get_attacked(G, arg)
                    set_out1.update(attackeds)
                set_und1 = args - set_in1 - set_out1
                diff2 = set_out1 & set_und

                print("======== STEP JUSTIFICATION =========", file=f_write)
                sp = f"We can add {diff1} to IN set: {set_in1} and add {diff2} to OUT set: {set_out1}, the rest is UNDEC set: {set_und1}. Let’s confirm this is a complete labelling."
                # if random.random() > 0.5:
                #     sp += "Because IN set is conflict-free. Furthermore: "
                # else:
                #     sp += "Because there is no attack relation in IN set. Furthermore: "
                print(sp, file=f_write)


                print(
                    f"(1)To confirm the arguments in IN set are legally labelled, we verify the attackers of {diff1} is labelled OUT.",
                    file=f_write)

                for i, arg in enumerate(diff1):
                    attackers = get_attacker(G, arg)
                    if len(attackers) == 1:
                        arg1 = list(attackers)[0]
                        print(f"(1.{i + 1}) {arg} is attacked by {arg1} while {arg1} is labelled OUT;",
                              file=f_write)

                        # attackers1 = get_attacker(G, arg1)
                        # att_options = []
                        # for att in attackers1:
                        #     if att in set_in1:
                        #         att_options.append(att)
                        # att = random.choice(att_options)
                        # print(f"{arg1} is attacked by {att} which is labelled IN.", file=f_write)

                    else:
                        print(f"(1.{i + 1}) {arg} is attacked by {attackers} while they are labelled OUT;",
                              file=f_write)
                        # str_attack = ""
                        # for arg1 in attackers:
                        #     attackers1 = get_attacker(G, arg1)
                        #     att_options = []
                        #     for att in attackers1:
                        #         if att in set_in1:
                        #             att_options.append(att)
                        #     att = random.choice(att_options)
                        #     str_attack += f"{arg1} is attacked by {att} which is labelled IN; "
                        # print(str_attack[:-2] + '.', file=f_write)
                print(
                    f"(2)To confirm the arguments in OUT set are legally labelled, we verify one of the attackers of {diff2} is labelled IN.",
                    file=f_write)
                for i, arg in enumerate(diff2):
                    attackers = get_attacker(G, arg)
                    for attacker in attackers:
                        if attacker in e_com:
                            print(
                                f"(2.{i + 1}) {arg} is attacked by {attackers} which is labelled IN;",
                              file=f_write)
                            break

                if len(set_und1) == 0:
                    print(f"(3)UNDEC set is empty.", file=f_write)
                else:
                    att_options = set()
                    for arg in set_und1:
                        attackers1 = get_attacker(G, arg)
                        att_options = att_options.union(attackers1)
                    print(
                        f"(3)The attackers of UNDEC set is {att_options}, while they are not all labelled OUT, and no attacker is labelled IN. ",
                        file=f_write)
                print(
                    f"Based on above assertions, we conclude this is a complete labelling. ",
                    file=f_write)
                print("======== STEP END =========", file=f_write)
                set_ins.append(set_in1)
                set_outs.append(set_out1)
                set_unds.append(set_und1)

        answer_section_com(set_ins, set_outs, set_unds, f_write)


def answer_section_com(set_ins, set_outs, set_unds, f_write):
    # ---------- answer
    set_ins = [sorted(list(s)) for s in set_ins]
    set_outs = [sorted(list(s)) for s in set_outs]
    set_unds = [sorted(list(s)) for s in set_unds]

    ans_dict = {"IN": set_ins, "OUT": set_outs, "UNDEC": set_unds}
    json_string = f"\nSo finally the answer in json format is {json.dumps(ans_dict, cls=JsonEncoder, indent=4)}"
    print(json_string, file=f_write)


def process_file(f, json_dir):
    print(f)
    with open(f, 'rb') as file:
        f_data = pickle.load(file)

    G = f_data['graph']
    exts = f_data['extensions']
    # draw_graph(G, save=True)

    grd_section(G, exts, json_dir, f)
    com_section(G, exts, json_dir, f)


def process_dataset(dataset_name):
    af_dir = config.dataset_dir / dataset_name / "AFs"
    filenames = list(af_dir.glob("*.pkl"))
    json_dir = config.dataset_dir / dataset_name / "prompt" / "txt"
    json_dir.mkdir(parents=True, exist_ok=True)

    with Pool(cpu_count()) as pool:
        list(tqdm(pool.starmap(process_file, [(f, json_dir) for f in filenames]), total=len(filenames)))


if __name__ == '__main__':
    random.seed(42)  # for reproducibility
    dataset_names = [f"test-{i}" for i in range(16, 26)]

    for dataset_name in dataset_names:
        process_dataset(dataset_name)
