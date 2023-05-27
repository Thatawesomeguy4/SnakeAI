import matplotlib.pyplot as plt
from IPython import display

plt.ion()
figure, axis = plt.subplots(2, sharex=True)

def plot(scores, mean_scores, time, mean_time):
    #display.clear_output(wait=True)
    #display(plt.gcf())
    #plt.clf()
    figure.suptitle('Training Data...')
    axis[0].plot(scores)
    axis[0].plot(mean_scores)
    axis[0].set(ylabel='Score')

    #plt.title('Training...')
    #plt.xlabel('Number of Games')
    #plt.ylabel('Score')
    #plt.plot(scores)
    #plt.plot(mean_scores)
    #plt.ylim(ymin=0)
    #plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    #plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    #plt.show()

    # plot time
    axis[1].plot(time)
    axis[1].plot(mean_time)
    axis[1].set(xlabel='Number of Games', ylabel='Time')
    #plt.clf()
    #plt.title('Time Chart')
    #plt.xlabel('Number of Games')
    #plt.ylabel('Time')
    #plt.plot(time)
    #plt.plot(mean_time)
    #plt.ylim(ymin=0)
    #plt.text(len(time) - 1, time[-1], str(time[-1]))
    #plt.text(len(mean_time) - 1, mean_time[-1], str(mean_time[-1]))
    plt.show()
    plt.pause(.1)
