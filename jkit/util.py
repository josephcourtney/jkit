
import itertools


def chunked_iterator(ary, chunk_shape, overlap):
    slices = [list() for _ in range(len(chunk_shape))]
    for i in range(len(chunk_shape)):
        for j in range(0, ary.shape[i], chunk_shape[i]-overlap[i]):
            slices[i].append(slice(j, min(ary.shape[i]-1, j+chunk_shape[i])))
    for chnk in itertools.product(*slices):
        yield tuple([slc.start for slc in chnk]), ary[chnk]


def random_pop(l):
    assert type(l) is list
    i = np.random.randint(len(l))
    e = l[i]
    l.pop(i)
    return e
