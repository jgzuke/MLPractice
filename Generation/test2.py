import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import random
class Game(object):
    
    def __init__(self, state):
        self.state = state

    def step(self, count = 1):
        for generation in range(count):
            new_board = [[False] * self.state.width for row in range(self.state.height)]

            for y, row in enumerate(self.state.board):
                for x, cell in enumerate(row):
                    neighbours = self.neighbours(x, y)
                    previous_state = self.state.board[y][x]
                    should_live = neighbours == 3 or (neighbours == 2 and previous_state == True)
                    new_board[y][x] = should_live

            self.state.board = new_board

    def neighbours(self, x, y):
        count = 0
        for hor in [x-1, x, x+1]:
            for ver in [y-1, y, y+1]:
                if (hor != x or ver != y) and (0 <= hor < self.state.width and 0 <= ver < self.state.height):
                    count += self.state.board[ver][hor]
        return count

    def display(self):
        return self.state.board

class State(object):
    
    def __init__(self, width, height, board = None):
        if not board:
            self.board = [[random.getrandbits(1) for x in range(width)] for y in range(height)]
        self.width = width
        self.height = height

my_game = Game(State(width = 50, height = 50))
fig = plt.figure()
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
image, = ax.imshow([], cmap='gray', interpolation='none')

def animate(i):
    image.set_data(my_game.display())
    my_game.step(1)
    return image


anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=4, interval=20)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save('basic_animation.mp4', fps=2, extra_args=['-vcodec', 'libx264'])

plt.show()