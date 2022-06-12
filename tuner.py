from sklearn.model_selection import GridSearchCV


class Tuner:
    @staticmethod
    def tune(model, param_grid, x, y):
        # fit
        grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=4, cv=3)
        grid_result = grid.fit(x, y)

        # show results
        print("Best result:", grid_result.best_score_, "params:", grid_result.best_params_)
        means = grid_result.cv_results_['mean_test_score']
        stds = grid_result.cv_results_['std_test_score']
        params = grid_result.cv_results_['params']
        print('mean_test_score', 'std_test_score', 'params')
        for mean, std, param in zip(means, stds, params):
            print(mean, std, param)
