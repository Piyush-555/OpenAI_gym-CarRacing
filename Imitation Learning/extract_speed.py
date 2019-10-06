# Import numpy
import numpy as np


# Method to extract speed from Bar-Plot on in-game screen
def get_speed(img):
    BLUE = np.array([0, 0, 255], dtype=np.float16)
    speedometer = img[93:83:-1, 19]
    
    speed = 0
    for i in speedometer:
        if np.array_equal(i, BLUE):
            speed+=1
        else:
            return speed


# Combining all things together
def main():
    data = np.load('../Data/frames.npy')
    frames = data.shape[0]
    speed = np.empty((frames, 1))

    for frame in range(frames):
        img = data[frame]
        speed[frame] = get_speed(img)

    np.save('../Data/speed.npy', speed)


if __name__ == '__main__':
    main()
