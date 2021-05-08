import matplotlib.pyplot as plt
import pickle

SCORES_PATH = "dqn_model_scores"

def load_scores(path):
    scores_file = open(path,'rb')
    scores = pickle.load(scores_file)
    scores_file.close()

    return scores

def scores_vs_episodes_plot(scores):
    episodes = [i for i in range(len(scores))]

    axes = plt.gca()
    axes.set_title("Scores vs Episodes")
    axes.set_xlabel("Episode #")
    axes.set_ylabel("Score")

    plt.plot(episodes, scores)
    


    plt.show()


if '__main__' == __name__:
    
    scores = load_scores(SCORES_PATH)
    scores_vs_episodes_plot(scores)