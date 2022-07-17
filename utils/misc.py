import random
import numpy as np
import pickle
import os


def random_col(seed=42, rng=(0, 1)):

    col = np.array([None, None, None])
    used = []
    for i, val in enumerate(col):
        col[i] = round(random.uniform(rng[0], rng[1]), 1)
        while col[i] in used:
            col[i] = round(random.uniform(0, 1), 1)
        used.append(col[i])

    return col


def random_colours(n, seed=42, min_diff=0.7):

    colours = []
    for i in range(n):
        col = random_col(seed=seed)
        if colours != []:    
            diffs = []
            for j, _ in enumerate(colours):
                diff = np.subtract(col, colours[j])
                diff = sum([abs(val) for val in diff])
                diffs.append(diff)
            while any(diff < min_diff for diff in diffs):
                col = random_col()
                diffs = []
                for j, _ in enumerate(colours):
                    diff = np.subtract(col, colours[j])
                    diff = sum([abs(val) for val in diff])
                    diffs.append(diff)
        colours.append(tuple(col))
    colours = [np.array(col) for col in colours]

    return colours


def get_items(names, o_path, func, *args, **kwargs):
    
    if type(names) == str:
        names = [names]

    if len(names) <= 1:
        name = names[0]    
        if f'{name}.pickle' not in os.listdir(o_path):
            item = func(*args, **kwargs)
            with open(f'{o_path}/{name}.pickle', 'wb') as f_name:
                pickle.dump(item, f_name)
        else:
            with open(f'{o_path}/{name}.pickle', 'rb') as f_name:
                item = pickle.load(f_name)
    
        return item

    else:
        item_names = [f'{name}.pickle' for name in names]
        if not all(item in os.listdir(o_path) for item in item_names):
            items = func(*args, **kwargs)  
            for i, _ in enumerate(items):
                name = item_names[i]
                with open(f'{o_path}/{name}', 'wb') as f_name:
                    pickle.dump(items[i], f_name)
        else:
            items = []
            for name in item_names:
                with open(f'{o_path}/{name}', 'rb') as f_name:
                    items.append(pickle.load(f_name))
            items = tuple(items)
    
        return items


def chunker(lst, num_chunks):

    length = len(lst)
    interval = int(length / num_chunks)
    chunks = []
    for i in range(0, length, interval):
        difference = length - (i + interval)
        if difference >= interval:
            chunk = lst[i: i+interval]
            chunks.append(chunk)
        else:
            chunk = lst[i:]
            chunks.append(chunk)
            break
  
    return chunks


def main():

    print(random_colours(13))


if __name__ == '__main__':
    main()


