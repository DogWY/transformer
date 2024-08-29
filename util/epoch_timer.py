

def epoch_time(start_time, end_time):
    """
    计算每个epoch的运行时间
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return: 运行时间
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
