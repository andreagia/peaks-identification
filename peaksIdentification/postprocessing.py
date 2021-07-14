################################################
################################################
################################################

def sliding_avg(x, half_window_size = 2):
    assert isinstance(x, list)
    X_ = x
    def window_avg(idx, size=half_window_size):
        left_idx = max(idx - size, 0)
        right_idx = min(idx + size + 1, len(x))
        w = X_[left_idx:right_idx]
        # print(x)
        # print(w, " --> ", sum(w) / len(w))
        return sum(w) / len(w)
    pos = range(len(X_))
    rr = map(window_avg, pos)
    return list(rr)



