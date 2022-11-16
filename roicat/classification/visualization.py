import matplotlib.pyplot as plt
import seaborn as sns

def confusion_matrix(cm, **params):
    default_params = dict(
        annot=True,
        annot_kws={"size": 16},
        vmin=0.,
        vmax=1.,
        cmap=plt.get_cmap('gray')
    )
    for key in params:
        default_params[key] = params[key]
    sns.heatmap(cm, **default_params)