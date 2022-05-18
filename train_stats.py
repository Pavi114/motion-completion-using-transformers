from genericpath import exists
import pickle
from constants import LAFAN1_DIRECTORY, OUTPUT_DIRECTORY

from util.extract import get_train_stats

def save_stats():
    x_mean, x_std, _, _ = get_train_stats(LAFAN1_DIRECTORY, ['subject1', 'subject2', 'subject3', 'subject4'])

    with open(f'{OUTPUT_DIRECTORY}/stats.pkl', 'wb') as f:
        pickle.dump(
            {'x_mean': x_mean, 'x_std': x_std},
            f,
            protocol=pickle.HIGHEST_PROTOCOL
        )

def load_stats():
    p = f'{OUTPUT_DIRECTORY}/stats.pkl'

    if not exists(p):
        save_stats()

    with open(f'{OUTPUT_DIRECTORY}/stats.pkl', 'rb') as f:
        stats = pickle.load(f)

    return stats['x_mean'], stats['x_std']

if __name__ == '__main__':
    save_stats()