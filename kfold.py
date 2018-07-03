def test_model(model_fn, model_args, fit_args, compile_args, verbose=True, k_fold=3):
    """
    Test a model using k-fold validation, returning the average loss
    :param model_fn: function that returns a Keras model and accepts as arguments
    **model_args
    :param model_args: dictionary with arguments to pass to model_fn(**model_args)
    :param fit_args: dictionary with arguments to pass to model.fit(**fit_args)
    :param compile_args: dictionary with arguments to pass to model.compile(
    **compile_args)
    :param verbose: if True show output
    :param k_fold:
    :return:
    """
    history = []
    if verbose:
        fit_args['verbose'] = 0
    for i in range(k_fold):
        if verbose:
            print('Run %s/%s' % (i, k_fold))
        model_temp = model_fn(**model_args)
        model_temp.compile(**compile_args)
        history.append(model_temp.fit(**fit_args))
    val_loss = []
    val_metric = []
    for result in history:
        local_val_loss = result.history['val_loss']
        local_val_metric = result.history['val_metric_degrees_difference']
        val_loss.append(local_val_loss)
        val_metric.append(local_val_metric)
    return val_loss, val_metric
