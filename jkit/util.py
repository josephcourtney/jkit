from datetime import datetime, timedelta
import itertools
import tzlocal

def nop(*args, **kwargs) -> None:  # noqa: ARG001
    pass

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

def format_timedelta(td):
    days = td.days
    seconds = td.seconds + 1e-6 * td.microseconds
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    parts = []
    if days > 0:
        parts.append(f"{days:02.0f}")
    if hours > 0:
        parts.append(f"{hours:02.0f}")
    if minutes > 0:
        parts.append(f"{minutes:02.0f}")
    parts.append(f"{seconds:06.3f}")
    return ":".join(parts)

def now():
    return datetime.now(tz=tzlocal.get_localzone())


