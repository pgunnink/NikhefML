import numpy as np


def random_search(evaluation_fn, evaluation_fn_params, params):
    """
    Using a random parameter combination as given in params, evaluate the function
    evalution_fn(**evaluation_fn_params) and return the result
    :param evaluation_fn:
    :param evaluation_fn_params:
    :param params:
    :return:
    """
    local_params = {}
    for param in params:
        dist = params[param]
        random_idx = np.random.randint(0,len(dist))
        local_params[param] = dist[random_idx]
    local_ev_params = evaluation_fn_params
    local_ev_params['model_args'] = local_params
    score_loss, score_metric = evaluation_fn(**local_ev_params)
    local_params['result_loss'] = score_loss
    local_params['result_metric'] = score_metric

    return local_params
